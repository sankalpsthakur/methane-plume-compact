"""Edge export: BN folding, ONNX export, static int8 (QDQ) quantisation, onnxruntime scoring + latency."""
import copy, os, time
import numpy as np
import torch
import torch.nn as nn


def fuse_conv_bn(model):
    """Deep copy with every (Conv2d, BatchNorm2d) pair inside an nn.Sequential fused into one Conv2d(bias=True)."""
    m = copy.deepcopy(model).eval()

    def rec(mod):
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Sequential):
                layers, new, i = list(child.children()), [], 0
                while i < len(layers):
                    if i + 1 < len(layers) and isinstance(layers[i], nn.Conv2d) and isinstance(layers[i + 1], nn.BatchNorm2d):
                        new.append(torch.nn.utils.fusion.fuse_conv_bn_eval(layers[i], layers[i + 1])); i += 2
                    else:
                        new.append(layers[i]); i += 1
                setattr(mod, name, nn.Sequential(*new))
            rec(getattr(mod, name))
    rec(m)
    return m


def export_onnx(model, path, hw=512, in_ch=4, opset=17):
    model = fuse_conv_bn(model).eval()
    x = torch.rand(1, in_ch, hw, hw)
    try:
        torch.onnx.export(model, x, path, input_names=["x"], output_names=["logit"], opset_version=opset, dynamo=False)
    except Exception as e:  # legacy exporter unavailable -> dynamo exporter
        print("legacy onnx export failed:", repr(e)[:200], "-> dynamo exporter", flush=True)
        torch.onnx.export(model, x, path, input_names=["x"], output_names=["logit"], opset_version=opset, dynamo=True)
    return path


def quantize_int8(fp32_path, int8_path, calib_tiles, method="minmax", per_channel=True):
    """Static int8 QDQ (S8 activations, S8 weights). calib_tiles: iterable of (1,4,H,W) float32 arrays from TRAIN data only."""
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self, tiles): self.it = iter([{"x": t.astype(np.float32)} for t in tiles])
        def get_next(self): return next(self.it, None)

    cm = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile, "entropy": CalibrationMethod.Entropy}[method]
    extra = {"CalibPercentile": 99.999} if method == "percentile" else {}
    quantize_static(fp32_path, int8_path, Reader(calib_tiles), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=per_channel, calibrate_method=cm, extra_options=extra)
    return int8_path


def ort_session(path, threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def ort_logits(sess, x):
    return sess.run(None, {"x": x.astype(np.float32)})[0]


def latency_ms(sess, hw=512, in_ch=4, n=30, warm=5):
    x = np.random.rand(1, in_ch, hw, hw).astype(np.float32)
    for _ in range(warm): sess.run(None, {"x": x})
    ts = []
    for _ in range(n):
        t = time.perf_counter(); sess.run(None, {"x": x}); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(np.min(ts))
