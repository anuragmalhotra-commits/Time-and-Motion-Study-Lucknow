"""Paths and defaults. Everything can be overridden with environment variables, so no
code needs editing on a new computer."""
import os
import sys
from pathlib import Path

# Auto-inject virtual environment packages into sys.path
for _extra in [r"C:\repnetvenv\Lib\site-packages", r"C:\repnet_pkgs"]:
    if Path(_extra).exists() and _extra not in sys.path:
        sys.path.insert(0, _extra)

APP_DIR = Path(__file__).resolve().parent.parent


def _path(env, default):
    return Path(os.environ.get(env, str(default))).expanduser()


# Folders
VIDEO_DIR = _path("INSIGHT_VIDEO_DIR", APP_DIR / "videos")          # videos listed in the app
RESULTS_DIR = _path("INSIGHT_RESULTS_DIR", APP_DIR / "results")     # all outputs
UPLOAD_DIR = APP_DIR / "_uploads"                                   # browser uploads (purged after 24 h)
MODELS_DIR = _path("INSIGHT_MODELS_DIR", APP_DIR / "models")        # model files
THIRD_PARTY_DIR = APP_DIR / "third_party"                           # downloaded model code
LEGACY_PKGS_DIR = _path("INSIGHT_PKGS_DIR", r"C:\repnet_pkgs")      # the original set-up

MASTER_CSV = RESULTS_DIR / "master_results.csv"
VALIDATION_DIR = RESULTS_DIR / "validation"

# Model files (first existing candidate is used)
HAND_MODEL_CANDIDATES = [MODELS_DIR / "hand_landmarker.task", LEGACY_PKGS_DIR / "hand_landmarker.task"]
REPNET_TORCH_WEIGHTS_CANDIDATES = [MODELS_DIR / "repnet_pytorch_weights.pth",
                                   THIRD_PARTY_DIR / "RepNet-pytorch" / "checkpoints" / "pytorch_weights.pth"]
REPNET_TORCH_CODE_DIR = THIRD_PARTY_DIR / "RepNet-pytorch"
TRANSRAC_CODE_DIR = THIRD_PARTY_DIR / "TransRAC"
SWIN_WEIGHTS = MODELS_DIR / "swin_tiny_patch244_window877_kinetics400_1k.pth"
TRANSRAC_CKPT_DIR = MODELS_DIR / "transrac"          # put the TransRAC checkpoint (.pt/.pth/.tar) here
REPNET_TF_CKPT_CANDIDATES = [MODELS_DIR / "repnet_ckpt", Path.home() / "OneDrive" / "Documents" / "repnet_ckpt",
                             LEGACY_PKGS_DIR / "repnet_ckpt"]
REPNET_TF_CODE_CANDIDATES = [LEGACY_PKGS_DIR, Path.home() / "OneDrive" / "Documents", APP_DIR / "legacy"]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# Memory safety: maximum bytes of decoded frames held at once (default 2.5 GB)
MAX_FRAME_BYTES = int(float(os.environ.get("INSIGHT_MAX_FRAME_GB", "2.5")) * 1e9)

# Study defaults (record any change - they are saved with every result)
DEFAULT_MIN_PAUSE_SEC = 2.0
ZERO_VIDEO_STORAGE = os.environ.get("INSIGHT_ZERO_VIDEO_STORAGE", "1") == "1"


def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def ensure_dirs():
    for d in (VIDEO_DIR, RESULTS_DIR, UPLOAD_DIR, MODELS_DIR, VALIDATION_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_dir_size_mb(path):
    p = Path(path)
    if not p.exists():
        return 0.0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)


def get_storage_stats():
    return {
        "uploads_mb": get_dir_size_mb(UPLOAD_DIR),
        "videos_mb": get_dir_size_mb(VIDEO_DIR),
        "results_mb": get_dir_size_mb(RESULTS_DIR),
        "models_mb": get_dir_size_mb(MODELS_DIR),
    }


def purge_all_uploads():
    """Removes all ephemeral uploaded videos to reclaim disk space immediately."""
    if not UPLOAD_DIR.exists():
        return 0
    count = 0
    for f in UPLOAD_DIR.iterdir():
        try:
            if f.is_file():
                f.unlink()
                count += 1
        except OSError:
            pass
    return count

