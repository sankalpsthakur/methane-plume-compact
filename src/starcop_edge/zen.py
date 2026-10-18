"""Zenodo (record 7863343) helpers: read zip members straight from the official archives with HTTP range requests.

Rate-limit policy (added after bench-gas-prep-train v2: 12 of 573 tiles ended with HTTP 429 after the old 8 x 1.5 s linear retry):
  * a shared semaphore caps the number of in-flight requests,
  * a 429 sets a pool-wide cool-down (Retry-After if the server sends it, else exponential), so every thread pauses together
    instead of hammering the endpoint with retries,
  * 5xx and network errors back off exponentially with jitter, permanent 4xx errors fail at once,
  * counters in STATS go into prep_report.json.
"""
import io, random, struct, threading, time, urllib.error, urllib.request, zipfile, zlib

REC = "https://zenodo.org/records/7863343/files/{}?download=1"

_SEM = threading.BoundedSemaphore(12)
_LOCK = threading.Lock()
_COOL = [0.0]  # monotonic-ish deadline (time.time()) before which no thread issues a request
STATS = {"requests": 0, "ok": 0, "http429": 0, "http5xx": 0, "other_err": 0, "cooldown_sleep_s": 0.0, "max_retry_after_s": 0.0}
PERMANENT = {400, 401, 403, 404, 410, 416}


def set_concurrency(n):
    """Cap the number of simultaneous requests (call before starting threads)."""
    global _SEM
    _SEM = threading.BoundedSemaphore(int(n))


def _bump(key, v=1):
    with _LOCK:
        STATS[key] += v


def _retry_after(e, k):
    h = getattr(e, "headers", None)
    ra = h.get("Retry-After") if h is not None else None
    try:
        return max(1.0, min(300.0, float(ra)))
    except (TypeError, ValueError):
        return min(180.0, 8.0 * 2 ** min(k, 5))


def _wait_cooldown():
    w = _COOL[0] - time.time()
    if w > 0:
        _bump("cooldown_sleep_s", w)
        time.sleep(w + random.uniform(0.0, 5.0))


def _range(url, start, end, tries=12, method="GET"):
    last = None
    for k in range(tries):
        _wait_cooldown()
        try:
            headers = {"Range": f"bytes={start}-{end}"} if method == "GET" else {}
            req = urllib.request.Request(url, headers=headers, method=method)
            _bump("requests")
            with _SEM:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read() if method == "GET" else int(r.headers["Content-Length"])
            _bump("ok")
            return data
        except urllib.error.HTTPError as e:
            last = e
            if e.code in PERMANENT:
                break
            if e.code == 429:
                d = _retry_after(e, k) * random.uniform(1.0, 1.25)
                _bump("http429")
                with _LOCK:
                    STATS["max_retry_after_s"] = max(STATS["max_retry_after_s"], d)
                    _COOL[0] = max(_COOL[0], time.time() + d)
            else:
                _bump("http5xx")
                time.sleep(min(120.0, 4.0 * 2 ** min(k, 5)) * random.uniform(1.0, 1.25))
        except Exception as e:  # timeouts, resets, short reads
            last = e
            _bump("other_err")
            time.sleep(min(60.0, 2.0 * 2 ** min(k, 5)) * random.uniform(1.0, 1.25))
    raise IOError(f"range fetch failed {url} {start}-{end}: {last!r}")


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile to read the central directory)."""

    def __init__(self, url, block=8 << 20):
        self.url, self.pos, self.block = url, 0, block
        self.size = _range(url, 0, 0, method="HEAD")
        self._cache = (0, b"")

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b""
        cs, cd = self._cache
        if cs <= self.pos and self.pos + n <= cs + len(cd):
            out = cd[self.pos - cs: self.pos - cs + n]
        else:
            fetch_n = max(n, self.block); start = self.pos
            if start + fetch_n > self.size: start = max(0, self.size - fetch_n)
            end = min(self.size - 1, start + fetch_n - 1)
            data = _range(self.url, start, end); self._cache = (start, data)
            out = data[self.pos - start: self.pos - start + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def archive_url(name):
    return REC.format(name if name.endswith(".zip") else name + ".zip")


def open_archive(name):
    return zipfile.ZipFile(HTTPRangeFile(archive_url(name)))


def read_member(url, zinfo, size_total):
    """Fetch ONE member (compressed bytes via a single range request), decompress, return raw bytes."""
    start = zinfo.header_offset
    end = min(size_total - 1, start + 30 + 1024 + zinfo.compress_size)
    blob = _range(url, start, end)
    assert blob[:4] == b"PK\x03\x04", "bad local header"
    nlen, elen = struct.unpack("<HH", blob[26:30])
    data = blob[30 + nlen + elen: 30 + nlen + elen + zinfo.compress_size]
    raw = zlib.decompress(data, -15) if zinfo.compress_type == 8 else data
    assert len(raw) == zinfo.file_size, "size mismatch"
    return raw
