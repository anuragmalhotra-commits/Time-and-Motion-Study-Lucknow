"""Downloads the model files the engines need (safe to run again: existing files are kept).

  python setup_models.py

Used by Setup_Environment.bat on Windows and by the Colab notebook. Every file comes from
its authors' own page; nothing is bundled with the app.
"""
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from tmbench import config

DOWNLOADS = [
    ("MediaPipe hand model", config.MODELS_DIR / "hand_landmarker.task",
     "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"),
    ("RepNet weights (PyTorch, ~100 MB)", config.MODELS_DIR / "repnet_pytorch_weights.pth",
     "https://huggingface.co/materight/repnet/resolve/main/pytorch_weights.pth"),
    ("Video Swin weights for TransRAC (~130 MB)", config.SWIN_WEIGHTS,
     "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth"),
]
CODE = [
    ("RepNet PyTorch code", config.REPNET_TORCH_CODE_DIR, "repnet/model.py",
     "https://codeload.github.com/materight/RepNet-pytorch/zip/refs/heads/main", "RepNet-pytorch-main"),
    ("TransRAC code", config.TRANSRAC_CODE_DIR, "models/TransRAC.py",
     "https://codeload.github.com/SvipRepetitionCounting/TransRAC/zip/refs/heads/main", "TransRAC-main"),
]
TRANSRAC_CKPT_HELP = (
    "The TransRAC checkpoint must be downloaded by hand (its authors share it on OneDrive/Baidu):\n"
    "  1. Open https://github.com/SvipRepetitionCounting/TransRAC and find 'The model ckpt has been available'.\n"
    "  2. Download the checkpoint from the OneDrive link (extraction code: transrac).\n"
    f"  3. Put the file in: {config.TRANSRAC_CKPT_DIR}\n")


def fetch(url, dest):
    tmp = Path(str(dest) + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "InSight-TimeMotion-Bench"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1024 * 1024)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def main():
    config.ensure_dirs()
    config.THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)
    config.TRANSRAC_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    problems = 0

    legacy_hand = config.LEGACY_PKGS_DIR / "hand_landmarker.task"
    target_hand = config.MODELS_DIR / "hand_landmarker.task"
    if not target_hand.exists() and legacy_hand.exists():
        shutil.copy2(legacy_hand, target_hand)
        print(f"[OK]   MediaPipe hand model copied from {legacy_hand}")

    for label, dest, url in DOWNLOADS:
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[OK]   {label} (already present)")
            continue
        print(f"[...]  Downloading {label}")
        try:
            fetch(url, dest)
            print(f"[OK]   {label}")
        except Exception as e:
            problems += 1
            print(f"[!]    {label}: download failed ({e}).\n       Download it from {url}\n       and save it as {dest}")

    for label, dest, marker, url, inner in CODE:
        if (dest / marker).exists():
            print(f"[OK]   {label} (already present)")
            continue
        print(f"[...]  Downloading {label}")
        zpath = config.THIRD_PARTY_DIR / f"{dest.name}.zip"
        try:
            fetch(url, zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(config.THIRD_PARTY_DIR)
            if dest.exists():
                shutil.rmtree(dest)
            (config.THIRD_PARTY_DIR / inner).rename(dest)
            print(f"[OK]   {label}")
        except Exception as e:
            problems += 1
            print(f"[!]    {label}: download failed ({e}). Download {url}, unzip it into "
                  f"{config.THIRD_PARTY_DIR} and rename the folder to {dest.name}.")
        finally:
            if zpath.exists():
                zpath.unlink()

    ck = [p for p in config.TRANSRAC_CKPT_DIR.iterdir() if p.suffix.lower() in (".pt", ".pth", ".tar", ".ckpt")]
    if ck:
        print(f"[OK]   TransRAC checkpoint ({ck[0].name})")
    else:
        print("[i]    " + TRANSRAC_CKPT_HELP.replace("\n", "\n       ").rstrip())
    print("\nDone." + (f" {problems} item(s) need attention - see above." if problems else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
