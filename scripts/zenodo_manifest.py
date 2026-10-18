"""Read the zip central directories of the official STARCOP Zenodo record (10.5281/zenodo.7863343)
using HTTP range requests (a few MB per archive, the data itself is NOT downloaded).
Writes manifests/zenodo_<archive>.tsv with: member name, uncompressed size, crc32 (hex), compress type."""
import io, sys, urllib.request, zipfile, time

REC = "https://zenodo.org/records/7863343/files/{}?download=1"
ARCHIVES = ["STARCOP_test.zip", "STARCOP_train_easy.zip"] + [f"STARCOP_train_remaining_part{i}.zip" for i in range(1, 6)]

class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20):
        self.url, self.pos, self.block = url, 0, block
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self._cache = (0, b"")
        self.nreq = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def _fetch(self, start, end):
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                self.nreq += 1
                return data
            except Exception as e:
                time.sleep(2 + 2 * attempt)
        raise IOError("range fetch failed")
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b""
        cs, cd = self._cache
        if cs <= self.pos and self.pos + n <= cs + len(cd):
            out = cd[self.pos - cs: self.pos - cs + n]
        else:
            fetch_n = max(n, self.block)
            start = self.pos
            if self.pos + fetch_n > self.size: start = max(0, self.size - fetch_n)
            end = min(self.size - 1, start + fetch_n - 1)
            data = self._fetch(start, end)
            self._cache = (start, data)
            out = data[self.pos - start: self.pos - start + n]
        self.pos += len(out)
        return out
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "manifests"
    for name in (sys.argv[2:] or ARCHIVES):
        f = HTTPRangeFile(REC.format(name), block=8 << 20)
        zf = zipfile.ZipFile(f)
        infos = [i for i in zf.infolist() if not i.is_dir()]
        with open(f"{outdir}/zenodo_{name[:-4]}.tsv", "w") as fh:
            for i in infos:
                fh.write(f"{i.filename}\t{i.file_size}\t{i.CRC:08x}\t{i.compress_type}\n")
        tot = sum(i.file_size for i in infos)
        print(f"{name}: archive {f.size/1e9:.3f} GB, {len(infos)} files, {tot/1e9:.3f} GB uncompressed, {f.nreq} range requests", flush=True)
