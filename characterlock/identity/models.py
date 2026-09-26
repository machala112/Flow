"""Model weight management for the face pipeline.

InsightFace models (buffalo_l pack) auto-download into
~/.insightface/models/ on first use. The inswapper model is NOT shipped by
InsightFace, so we fetch it from a community mirror into the same directory
and verify its SHA256 before use.

inswapper_128.onnx has no clear license; InsightFace directs commercial
users to contact contact@insightface.ai. We download at runtime (never
redistribute weights) — see README / THIRD_PARTY_NOTICES.
"""
import hashlib
import os
import urllib.request

INSWAPPER_NAME = "inswapper_128.onnx"
# Published by the community (see research notes); re-verify if a mirror
# starts serving a different file.
INSWAPPER_SHA256 = (
    "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af")
INSWAPPER_URLS = [
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/"
    "inswapper_128.onnx",
    "https://huggingface.co/thebiglaskowski/inswapper_128.onnx/resolve/main/"
    "inswapper_128.onnx",
    "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/"
    "inswapper_128.onnx",
]
BUFFALO_PACK = "buffalo_l"


def models_dir():
    """Directory InsightFace uses for weights (respects INSIGHTFACE_HOME)."""
    base = os.environ.get("INSIGHTFACE_HOME",
                          os.path.join(os.path.expanduser("~"), ".insightface"))
    d = os.path.join(base, "models")
    os.makedirs(d, exist_ok=True)
    return d


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dest, progress=None):
    req = urllib.request.Request(url, headers={"User-Agent": "CharacterLock"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0) or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)


def ensure_inswapper(progress=None):
    """Return path to a hash-verified inswapper_128.onnx, downloading it."""
    dest = os.path.join(models_dir(), INSWAPPER_NAME)
    if os.path.isfile(dest) and _sha256(dest) == INSWAPPER_SHA256:
        return dest
    errors = []
    for url in INSWAPPER_URLS:
        try:
            if progress:
                progress(f"downloading inswapper_128.onnx from {url}")
            _download(url, dest + ".part",
                      progress=lambda d, t: progress(
                          f"inswapper_128.onnx: {d/1e6:.0f}/{t/1e6:.0f} MB")
                      if progress else None)
            if _sha256(dest + ".part") != INSWAPPER_SHA256:
                raise ValueError("SHA256 mismatch")
            os.replace(dest + ".part", dest)
            return dest
        except Exception as e:  # try next mirror
            errors.append(f"{url}: {e}")
            if os.path.isfile(dest + ".part"):
                os.unlink(dest + ".part")
    raise RuntimeError("could not download inswapper_128.onnx:\n" +
                       "\n".join(errors))


def get_swapper():
    """Load the inswapper model via InsightFace's model zoo."""
    import insightface
    path = ensure_inswapper()
    # InsightFace 2.0 treats a name ending in .onnx as a literal file path
    # (it does not join it with the model root), so pass the absolute path.
    return insightface.model_zoo.get_model(path)
