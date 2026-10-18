"""Models: (1) EdgeUNet, a compact plain U-Net family; (2) SMPLikeUnet, a dependency-free re-implementation of
segmentation_models_pytorch.Unet('mobilenet_v2') whose state_dict keys match the released HyperSTARCOP checkpoint
(after stripping the 'network.' prefix)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------------
# compact edge U-Net
# --------------------------------------------------------------------------------------------------
def cbr(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class EdgeUNet(nn.Module):
    """Plain U-Net. Encoder level k>0 starts with a stride-2 3x3 conv; decoder uses nearest x2 upsampling + concat.
    widths: channels per level, e.g. (8, 16, 32, 64, 128). dec_widths defaults to widths[:-1] (decoder output per level).
    stem_stride=2: the first encoder conv is stride 2 (all levels run at half resolution, ~4x fewer MACs); the full-resolution
    detail is restored by bilinear upsampling + a 4-channel full-res shallow feature + a 3x3 output conv."""

    def __init__(self, in_ch=4, widths=(8, 16, 32, 64, 128), dec_widths=None, out_ch=1, stem_stride=1):
        super().__init__()
        self.widths = tuple(widths); self.stem_stride = stem_stride
        L = len(widths)
        dec_widths = tuple(dec_widths) if dec_widths is not None else tuple(widths[:-1])
        assert len(dec_widths) == L - 1
        self.enc = nn.ModuleList()
        c_prev = in_ch
        for k, w in enumerate(widths):
            st = stem_stride if k == 0 else 2
            self.enc.append(nn.Sequential(cbr(c_prev, w, stride=st), cbr(w, w)))
            c_prev = w
        self.dec = nn.ModuleList()  # index j handles level L-2-j
        c_up = widths[-1]
        for j in range(L - 1):
            k = L - 2 - j
            w = dec_widths[k]
            self.dec.append(nn.Sequential(cbr(c_up + widths[k], w), cbr(w, w)))
            c_up = w
        if stem_stride == 2:
            self.skip0 = cbr(in_ch, 4)
            self.head = nn.Conv2d(c_up + 4, out_ch, 3, 1, 1)
        else:
            self.head = nn.Conv2d(c_up, out_ch, 1)

    def forward(self, x):
        x_in = x
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
        x = skips.pop()
        for d in self.dec:
            s = skips.pop()
            x = F.interpolate(x, size=s.shape[-2:], mode="nearest")
            x = d(torch.cat([x, s], dim=1))
        if self.stem_stride == 2:
            s0 = self.skip0(x_in)
            x = F.interpolate(x, size=s0.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, s0], dim=1)
        return self.head(x)


# --------------------------------------------------------------------------------------------------
# smp-compatible MobileNetV2 U-Net (released HyperSTARCOP architecture)
# --------------------------------------------------------------------------------------------------
class ConvBNReLU6(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden = int(round(inp * expand_ratio))
        self.use_res = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(inp, hidden, k=1))
        layers += [ConvBNReLU6(hidden, hidden, stride=stride, groups=hidden),
                   nn.Conv2d(hidden, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV2Encoder(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        feats = [ConvBNReLU6(in_ch, 32, stride=2)]
        c = 32
        for t, oc, n, s in setting:
            for i in range(n):
                feats.append(InvertedResidual(c, oc, s if i == 0 else 1, t))
                c = oc
        feats.append(ConvBNReLU6(c, 1280, k=1))
        self.features = nn.ModuleList(feats)
        self.out_channels = (in_ch, 16, 24, 32, 96, 1280)

    def forward(self, x):
        f = self.features
        out = [x]
        for stage in (f[:2], f[2:4], f[4:7], f[7:14], f[14:]):
            for m in stage:
                x = m(x)
            out.append(x)
        return out


class Conv2dReLU(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.conv1 = Conv2dReLU(cin + skip, cout)
        self.conv2 = Conv2dReLU(cout, cout)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()
        enc = list(encoder_channels[1:])[::-1]  # (1280, 96, 32, 24, 16)
        head = enc[0]
        in_ch = [head] + list(decoder_channels[:-1])
        skip_ch = list(enc[1:]) + [0]
        self.blocks = nn.ModuleList([DecoderBlock(i, s, o) for i, s, o in zip(in_ch, skip_ch, decoder_channels)])

    def forward(self, feats):
        feats = feats[1:][::-1]
        x, skips = feats[0], feats[1:]
        for i, b in enumerate(self.blocks):
            x = b(x, skips[i] if i < len(skips) else None)
        return x


class SMPLikeUnet(nn.Module):
    """== smp.Unet('mobilenet_v2', encoder_weights=None, in_channels=in_ch, classes=1, activation=None)."""

    def __init__(self, in_ch=4, classes=1):
        super().__init__()
        self.encoder = MobileNetV2Encoder(in_ch)
        self.decoder = UnetDecoder(self.encoder.out_channels)
        self.segmentation_head = nn.Sequential(nn.Conv2d(16, classes, 3, 1, 1))

    def forward(self, x):
        return self.segmentation_head(self.decoder(self.encoder(x)))


def load_released_checkpoint(path, map_location="cpu"):
    """Load the released Lightning checkpoint without needing pytorch_lightning/omegaconf installed."""
    import pickle

    class _Dummy:
        def __init__(self, *a, **k): pass
        def __setstate__(self, s): self.state = s

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return type(name, (_Dummy,), {})

    class _PM:
        Unpickler = _Unpickler
        load = staticmethod(pickle.load)
        loads = staticmethod(pickle.loads)

    ck = torch.load(path, map_location=map_location, pickle_module=_PM, weights_only=False)
    sd = {k[len("network."):]: v for k, v in ck["state_dict"].items() if k.startswith("network.")}
    return sd, ck


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


# input normalisation constants: identical to starcop.data.normalizer_module.BAND_NORMALIZATION
INPUT_ORDER = ("mag1c", "TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm")
INPUT_FACTORS = (1750.0, 60.0, 60.0, 60.0)
INPUT_CLIP = (0.0, 2.0)
