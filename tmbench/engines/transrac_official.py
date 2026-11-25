"""Official TransRAC (Hu et al., CVPR 2022), run without the old mmcv/mmaction libraries.

The official model code (github.com/SvipRepetitionCounting/TransRAC, Apache 2.0) is used
unchanged; only three import lines that need mmcv/mmaction are swapped for stand-ins and
weights are loaded with plain PyTorch. Needs: the TransRAC code and Video Swin weights
(downloaded by Setup_Environment.bat) and the TransRAC checkpoint (manual download).

TransRAC looks at 64 frames at a time. Each window of `window_sec` seconds is sampled into
64 frames; the model's per-frame density is summed. Window length decides how many frames
each cycle gets: in TransRAC's training videos a repetition usually spans several of the 64
frames, so very short cycles need shorter windows.
"""
import importlib.util
import math
import os
import types
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .. import config
from .base import Engine, EngineResult, Setting, report

SWIN_TINY_CFG = dict(patch_size=(2, 4, 4), embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                     window_size=(8, 7, 7), mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                     attn_drop_rate=0.0, drop_path_rate=0.1, patch_norm=True)
_CACHE = {}


class _Registry:
    def register_module(self, *a, **k):
        return lambda cls: cls


def _exec_patched(path, replacements, name, extra=None):
    code = Path(path).read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in code:
            raise RuntimeError(f"Unexpected TransRAC source: '{old}' not found in {path}.")
        code = code.replace(old, new)
    mod = types.ModuleType(name)
    mod.__dict__.update(extra or {})
    exec(compile(code, str(path), "exec"), mod.__dict__)
    return mod


def find_checkpoint():
    d = config.TRANSRAC_CKPT_DIR
    if not d.exists():
        return None
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".pt", ".pth", ".tar", ".ckpt"))
    return files[0] if files else None


def load_transrac(repo_dir, swin_ckpt, transrac_ckpt, device):
    import torch
    repo = Path(repo_dir)
    swin = _exec_patched(repo / "mmaction/models/backbones/swin_transformer.py", [
        ("from mmcv.runner import load_checkpoint", "load_checkpoint = None"),
        ("from mmaction.utils import get_root_logger", "get_root_logger = None"),
        ("from ..builder import BACKBONES", "BACKBONES = _Registry()"),
    ], "transrac_swin", {"_Registry": _Registry})
    tr = _exec_patched(repo / "models/TransRAC.py", [
        ("from mmcv import Config", "Config = None"),
        ("from mmaction.models import build_model", "build_model = None"),
        ("from mmcv.runner import load_checkpoint", "load_checkpoint = None"),
    ], "transrac_model")

    def build_backbone(path):
        bb = swin.SwinTransformer3D(**SWIN_TINY_CFG)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        missing, _ = bb.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"Video Swin weights incomplete ({len(missing)} missing).")
        return bb

    class StandaloneTransRAC(tr.TransferModel):
        def load_model(self):
            return build_backbone(self.checkpoint)

    model = StandaloneTransRAC(config=None, checkpoint=str(swin_ckpt), num_frames=64, scales=[1, 4, 8], OPEN=False)
    info = {}
    if transrac_ckpt is not None:
        ck = torch.load(transrac_ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        own = model.state_dict()
        matched = {k: v for k, v in sd.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
        missing, _ = model.load_state_dict(matched, strict=False)
        head_missing = [k for k in missing if not k.startswith("backbone.")]
        if head_missing:
            raise RuntimeError(f"The TransRAC checkpoint is missing {len(head_missing)} counting-head weights "
                               f"(e.g. {head_missing[:2]}). Check that the right file is in {config.TRANSRAC_CKPT_DIR}.")
        info = {"checkpoint": Path(transrac_ckpt).name, "weights_matched": len(matched)}
    model.eval().to(device)
    return model, info


class TransRACEngine(Engine):
    id = "transrac"
    label = "TransRAC (official model)"
    kind = "trained model"
    short = "The published TransRAC repetition counter (CVPR 2022), trained on exercise videos."
    about = ("Official TransRAC: Video Swin features at three time scales, multi-scale self-similarity and a "
             "density map. Runs here without its original mmcv/mmaction libraries. Trained on RepCount-A "
             "(mostly exercise videos), not embroidery. Slow on a laptop CPU (roughly 2 s per second of video); "
             "fine for validation clips, a GPU is better for whole sessions. Not yet checked against manual counts "
             "on chikankari.")
    frame_size = 224
    target_fps = None
    settings = [
        Setting("window_sec", "Window length (s)", "float", 20.0,
                "Each window becomes 64 frames. Shorter windows suit faster cycles (each cycle should span "
                "about 4 or more of the 64 frames).", 5.0, 60.0, 1.0),
    ]

    def check(self):
        for pkg in ("torch", "timm", "einops"):
            if importlib.util.find_spec(pkg) is None:
                return False, f"The {pkg} package is not installed (run Setup_Environment.bat)."
        if not (config.TRANSRAC_CODE_DIR / "models" / "TransRAC.py").exists():
            return False, f"TransRAC code not found in {config.TRANSRAC_CODE_DIR} (Setup_Environment.bat downloads it)."
        if not config.SWIN_WEIGHTS.exists():
            return False, f"Video Swin weights not found ({config.SWIN_WEIGHTS.name}; Setup_Environment.bat downloads them)."
        if find_checkpoint() is None and os.environ.get("INSIGHT_TRANSRAC_RANDOM_HEAD_FOR_TESTING") != "1":
            return False, ("TransRAC checkpoint not found. Download it from the TransRAC GitHub page (OneDrive link, "
                           f"extraction code 'transrac') and put the file in {config.TRANSRAC_CKPT_DIR}.")
        return True, f"Ready (checkpoint: {find_checkpoint().name if find_checkpoint() else 'RANDOM HEAD - TEST MODE'})"

    def run(self, clip, settings, progress=None):
        import torch
        s = {**self.default_settings(), **(settings or {})}
        ok, msg = self.check()
        if not ok:
            raise RuntimeError(msg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        key = (str(find_checkpoint()), device)
        report(progress, 0.02, "Loading TransRAC...")
        if key not in _CACHE:
            _CACHE.clear()
            _CACHE[key] = load_transrac(config.TRANSRAC_CODE_DIR, config.SWIN_WEIGHTS, find_checkpoint(), device)
        model, info = _CACHE[key]

        win = float(s["window_sec"])
        n_win = max(1, math.ceil(clip.duration / win - 1e-6))
        cap = cv2.VideoCapture(str(clip.path))
        if not cap.isOpened():
            raise IOError(f"Cannot open {clip.path}")
        sample_t, sample_d, windows = [], [], []
        roi = clip.roi
        for w in range(n_win):
            a = clip.start_sec + w * win
            b = min(clip.end_sec, a + win)
            if b - a < 2.0:
                windows.append({"start_sec": round(a, 2), "end_sec": round(b, 2), "count": np.nan,
                                "note": "shorter than 2 s, not counted"})
                continue
            times = np.linspace(a, b, 64, endpoint=False)
            frames, last = [], None
            for t in times:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
                ok_r, f = cap.read()
                if not ok_r:
                    f = last
                if f is None:
                    break
                last = f
                f = roi.apply(f)
                frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (224, 224)))
            if len(frames) < 64:
                break
            x = torch.from_numpy(np.stack(frames)).float().div(255.0).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            with torch.no_grad():
                dens, _ = model(x)
            d_raw = dens.float().cpu().numpy().reshape(-1)
            c = float(d_raw.sum())                   # the window count, as in the official evaluation
            d_pos = np.clip(d_raw, 0, None)          # the raw density can dip slightly below zero, so
            d = d_pos * (max(c, 0.0) / d_pos.sum()) if d_pos.sum() > 0 else np.zeros_like(d_raw)
            sample_t.extend((times - clip.start_sec).tolist())   # the time line uses the positive part,
            sample_d.extend(d.tolist())                          # rescaled to the official count
            windows.append({"start_sec": round(a, 2), "end_sec": round(b, 2), "count": round(max(c, 0.0), 3),
                            "raw_count": round(c, 3)})
            report(progress, 0.05 + 0.9 * (w + 1) / n_win, f"TransRAC window {w + 1}/{n_win}")
        cap.release()
        if not sample_t:
            raise ValueError("No frames could be read for TransRAC.")

        # Put the counts on a regular 15 FPS time line, like the other engines
        fps = 15.0
        T = max(1, int(round(clip.duration * fps)))
        grid = np.arange(T) / fps
        st_ = np.asarray(sample_t)
        cum_samples = np.cumsum(sample_d)
        cum = np.interp(grid, st_, cum_samples, left=0.0, right=cum_samples[-1])
        rate = pd.Series(np.diff(np.concatenate([[0.0], cum]))).rolling(int(2 * fps), center=True, min_periods=1).mean().values
        ref = np.percentile(rate, 95) if T else 0.0
        active = rate > 0.15 * ref if ref > 0 else np.zeros(T, bool)
        warnings = []
        if find_checkpoint() is None:
            warnings.append("TEST MODE: random counting head - results are meaningless.")
        frames_per_cycle = None
        tot = float(np.nansum([w.get("count", 0) for w in windows]))
        if tot > 0:
            frames_per_cycle = 64 * len([w for w in windows if not np.isnan(w.get("count", np.nan))]) / tot
            if frames_per_cycle < 4:
                warnings.append(f"Each counted cycle spans only about {frames_per_cycle:.1f} of the 64 frames; "
                                "try a shorter window so cycles are not missed.")
        report(progress, 1.0, "TransRAC done")
        return EngineResult(self.id, fps, cum, active, score=None,
                            method_used=f"Official TransRAC, {win:.0f} s windows x 64 frames, {device}",
                            details={**info, "windows": len(windows), "device": device,
                                     "frames_per_cycle": round(frames_per_cycle, 2) if frames_per_cycle else None,
                                     "per_window": windows},
                            warnings=warnings)
