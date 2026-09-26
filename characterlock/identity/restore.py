"""Face restoration with GFPGAN (TencentARC, Apache-2.0).

After the inswapper identity swap, swapped faces can carry blending
artifacts. GFPGANv1.4 restores them and blends them into the frame's
lighting/skin tone. Weights (~349 MB) download on first use with a
HuggingFace mirror fallback — the official TencentARC release asset has
been flaky.
"""
import os
import urllib.request

WEIGHTS_URLS = [
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/"
    "GFPGANv1.4.pth",
    "https://huggingface.co/nlightcho/gfpgan_v14/resolve/main/GFPGANv1.4.pth",
    "https://huggingface.co/Apex-X/gfpgan.pth/resolve/main/GFPGANv1.4.pth",
]
WEIGHTS_NAME = "GFPGANv1.4.pth"
WEIGHTS_MIN_SIZE = 300_000_000  # ~349 MB sanity floor


def weights_path():
    d = os.path.join(os.path.expanduser("~"), ".characterlock", "models",
                     "gfpgan")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, WEIGHTS_NAME)


def ensure_weights(progress=None):
    """Download GFPGANv1.4.pth (if needed); return the local path."""
    dest = weights_path()
    if os.path.isfile(dest) and os.path.getsize(dest) >= WEIGHTS_MIN_SIZE:
        return dest
    errors = []
    for url in WEIGHTS_URLS:
        try:
            if progress:
                progress(f"downloading {WEIGHTS_NAME}")
            req = urllib.request.Request(url,
                                         headers={"User-Agent": "CharacterLock"})
            with urllib.request.urlopen(req) as r, \
                    open(dest + ".part", "wb") as f:
                done = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(f"{WEIGHTS_NAME}: {done/1e6:.0f} MB")
            if os.path.getsize(dest + ".part") < WEIGHTS_MIN_SIZE:
                raise ValueError("download too small, likely failed")
            os.replace(dest + ".part", dest)
            return dest
        except Exception as e:
            errors.append(f"{url}: {e}")
            if os.path.isfile(dest + ".part"):
                os.unlink(dest + ".part")
    raise RuntimeError("could not download GFPGANv1.4.pth:\n" +
                       "\n".join(errors))


def get_restorer(device=None, progress=None):
    """Build a GFPGANer (upscale=1, paste-back) for frame restoration."""
    from gfpgan.utils import GFPGANer
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return GFPGANer(
        model_path=ensure_weights(progress),
        upscale=1,            # keep input resolution
        arch="clean",         # v1.4 weights use the clean architecture
        channel_multiplier=2,
        bg_upsampler=None,    # no Real-ESRGAN background pass needed
        device=device,
    )


def restore_frame(restorer, frame_bgr, weight=0.5):
    """Restore faces in a BGR frame; paste back onto the original.

    Returns the restored BGR frame. If no face is found, returns the
    input unchanged.
    """
    _, restored_faces, restored_img = restorer.enhance(
        frame_bgr, has_aligned=False, only_center_face=False,
        paste_back=True, weight=weight)
    if restored_img is None:
        return frame_bgr
    return restored_img
