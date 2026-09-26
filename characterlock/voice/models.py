"""OpenVoice V2 checkpoint management.

We use only the V2 *converter* (~130 MB): config.json + checkpoint.pth from
the HuggingFace repo myshell-ai/OpenVoiceV2. Checkpoints live under
~/.characterlock/models/ and are downloaded on first use.

OpenVoice is MIT-licensed (myshell-ai/OpenVoice); see THIRD_PARTY_NOTICES.
"""
import os
import urllib.request

CONFIG_URL = ("https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/"
              "converter/config.json")
CKPT_URL = ("https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/"
            "converter/checkpoint.pth")
# Approximate sizes, used only to sanity-check downloads.
CONFIG_SIZE = 500
CKPT_SIZE = 126_000_000


def models_dir():
    d = os.path.join(os.path.expanduser("~"), ".characterlock", "models",
                     "openvoice_v2")
    os.makedirs(d, exist_ok=True)
    return d


def _fetch(url, dest, min_size, progress=None):
    if os.path.isfile(dest) and os.path.getsize(dest) >= min_size:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "CharacterLock"})
    with urllib.request.urlopen(req) as r, open(dest + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    if os.path.getsize(dest + ".part") < min_size:
        os.unlink(dest + ".part")
        raise RuntimeError(f"download too small, likely failed: {url}")
    os.replace(dest + ".part", dest)
    return dest


def ensure_checkpoints(progress=None):
    """Download (if needed) and return (config_path, ckpt_path)."""
    d = models_dir()
    cfg = _fetch(CONFIG_URL, os.path.join(d, "config.json"), CONFIG_SIZE,
                 progress)
    ckpt = _fetch(CKPT_URL, os.path.join(d, "checkpoint.pth"), CKPT_SIZE,
                  progress)
    return cfg, ckpt


def pick_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_converter(device=None, progress=None):
    """Build a ToneColorConverter with V2 weights loaded."""
    from openvoice.api import ToneColorConverter
    cfg, ckpt = ensure_checkpoints(progress)
    device = device or pick_device()
    # NB: ToneColorConverter.__init__ forwards **kwargs to its base class,
    # which rejects enable_watermark (upstream quirk), so we construct
    # normally and then disable the watermark model — CharacterLock output
    # must not carry an inaudible watermark.
    conv = ToneColorConverter(cfg, device=device)
    conv.watermark_model = None
    conv.load_ckpt(ckpt)
    return conv
