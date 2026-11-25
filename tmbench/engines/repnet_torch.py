"""Google RepNet (CVPR 2020), run through the PyTorch port by materight
(github.com/materight/RepNet-pytorch) with weights converted from Google's checkpoint.

The code and weights are downloaded on your computer by Setup_Environment.bat; they are
not bundled with this app.
"""
import importlib.util
import os
import sys

import numpy as np

from .. import config
from .base import Engine, EngineResult, Setting, report

_MODEL_CACHE = {}


class RepNetTorchEngine(Engine):
    id = "repnet_torch"
    label = "RepNet (PyTorch)"
    kind = "trained model"
    short = "Google's RepNet repetition counter, trained on everyday videos (not embroidery)."
    about = ("RepNet looks at 64 frames at a time, compares every frame with every other, and predicts the period "
             "and whether each frame is part of a repetition. Several temporal strides are tried and the most "
             "confident is used, unless you fix the stride (recommended for the study, so every session is analysed "
             "the same way). At 15 FPS and stride 1 it can see cycles of about 0.2-2.1 s; stride 2 doubles that. "
             "Trained on general videos, not embroidery; not yet checked against manual counts on chikankari.")
    frame_size = 112
    target_fps = 15
    fit = "squash"   # as in RepNet's own scripts
    settings = [
        Setting("strides", "Strides to try", "text", "1,2,3,4",
                "Comma-separated. One value fixes the stride (recommended once chosen from the validation sample)."),
        Setting("periodicity_threshold", "Periodicity threshold", "float", 0.5,
                "Frames below this are not counted (RepNet authors' default 0.5).", 0.05, 0.95, 0.05),
        Setting("analysis_fps", "Analysis frame rate", "choice", 15, "Frames per second given to RepNet.", options=[10, 15, 20, 30]),
    ]

    def _weights(self):
        p = config.first_existing(config.REPNET_TORCH_WEIGHTS_CANDIDATES)
        return str(p) if p else None

    def check(self):
        if importlib.util.find_spec("torch") is None:
            return False, "PyTorch is not installed (run Setup_Environment.bat)."
        if importlib.util.find_spec("timm") is None:
            return False, "The timm package is not installed (run Setup_Environment.bat)."
        if not (config.REPNET_TORCH_CODE_DIR / "repnet" / "model.py").exists():
            return False, f"RepNet code not found in {config.REPNET_TORCH_CODE_DIR} (Setup_Environment.bat downloads it)."
        if not self._weights() and os.environ.get("INSIGHT_REPNET_RANDOM_WEIGHTS_FOR_TESTING") != "1":
            return False, (f"RepNet weights not found. Expected {config.REPNET_TORCH_WEIGHTS_CANDIDATES[0]} "
                           "(Setup_Environment.bat downloads them).")
        return True, f"Ready (weights: {self._weights() or 'RANDOM - TEST MODE'})"

    def _load(self):
        import torch
        code = str(config.REPNET_TORCH_CODE_DIR)
        if code not in sys.path:
            sys.path.insert(0, code)
        from repnet.model import RepNet
        key = self._weights() or "random"
        if key not in _MODEL_CACHE:
            # RepNet's code fetches the ResNet50-V2 backbone through torch.hub, which downloads
            # code at run time and asks an interactive question (that hangs inside the app).
            # The same backbone comes from the installed timm package instead; the RepNet
            # weights loaded below replace all its parameters anyway.
            import timm
            original_hub_load = torch.hub.load
            torch.hub.load = lambda repo, name, *a, **k: timm.create_model(name, pretrained=False)
            try:
                model = RepNet()
            finally:
                torch.hub.load = original_hub_load
            if self._weights():
                state = torch.load(self._weights(), map_location="cpu", weights_only=True)
                model.load_state_dict(state)
            model.eval()
            _MODEL_CACHE.clear()
            _MODEL_CACHE[key] = model
        return _MODEL_CACHE[key], RepNet

    def run(self, clip, settings, progress=None):
        import torch
        s = {**self.default_settings(), **(settings or {})}
        ok, msg = self.check()
        if not ok:
            raise RuntimeError(msg)
        try:
            strides = sorted({int(x) for x in str(s["strides"]).replace(" ", "").split(",") if x})
        except ValueError as err:
            raise ValueError("Strides must be whole numbers separated by commas, e.g. 1,2,3,4") from err
        fps_req = int(s["analysis_fps"])
        report(progress, 0.05, "Loading RepNet...")
        model, RepNet = self._load()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        frames, fps, _ = clip.frames(fps_req, self.frame_size, self.fit)
        T = len(frames)
        if T < 64:
            raise ValueError("RepNet needs at least 64 frames (about 4 s at 15 FPS). Choose a longer segment.")

        best = None
        total_steps = sum(int(np.ceil(len(range(0, T, st)) / 64)) for st in strides)
        done = 0
        for st in strides:
            idx = np.arange(0, T, st)
            n_chunks = int(np.ceil(len(idx) / 64))
            if n_chunks == 0:
                continue
            pad = n_chunks * 64 - len(idx)
            idx = np.concatenate([idx, np.full(pad, idx[-1])])   # repeat the last frame to fill the final chunk
            raw_len, raw_per = [], []
            with torch.no_grad():
                for c in range(n_chunks):
                    sel = frames[idx[c * 64:(c + 1) * 64]].astype(np.float32) / 255.0
                    x = torch.from_numpy((sel - 0.5) / 0.5).permute(3, 0, 1, 2).unsqueeze(0).to(device)  # 1x3x64x112x112
                    pl, pr, _ = model(x)
                    raw_len.append(pl[0].cpu())
                    raw_per.append(pr[0].cpu())
                    done += 1
                    report(progress, 0.1 + 0.85 * done / max(1, total_steps), f"RepNet stride {st}: chunk {c + 1}/{n_chunks}")
            conf, period_len, period_count, per_score = RepNet.get_counts(
                torch.cat(raw_len), torch.cat(raw_per), st, periodicity_threshold=float(s["periodicity_threshold"]))
            cum = period_count.numpy()[:T]
            score = per_score.numpy()[:T]
            if len(cum) < T:   # should not happen, but keep arrays aligned
                cum = np.concatenate([cum, np.full(T - len(cum), cum[-1] if len(cum) else 0.0)])
                score = np.concatenate([score, np.zeros(T - len(score))])
            if best is None or float(conf) > best[0]:
                best = (float(conf), st, cum, score, float(period_len))
        conf, st, cum, score, period_len = best
        active = score >= float(s["periodicity_threshold"])
        warnings = []
        if len(strides) > 1:
            warnings.append(f"Stride chosen automatically ({st}). For the study, fix one stride so every session "
                            "is analysed the same way.")
        if not self._weights():
            warnings.append("TEST MODE: random weights - results are meaningless.")
        report(progress, 1.0, "RepNet done")
        return EngineResult(
            self.id, fps, cum, active, score=score,
            method_used=f"RepNet PyTorch port, stride {st}, {fps:.0f} FPS, threshold {s['periodicity_threshold']}",
            details={"chosen_stride": st, "confidence": round(conf, 3), "device": device,
                     "mean_period_frames": round(period_len, 2)},
            warnings=warnings)
