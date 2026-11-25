"""RepNet through the ORIGINAL TensorFlow set-up (your batch_repnet.py + repnet_ckpt),
called exactly as the earlier app called it. Useful to check that the new PyTorch RepNet
gives the same answers as the set-up you already used.

This adapter could not be tested by the developer (batch_repnet.py was not available).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from .. import config
from .base import Engine, EngineResult, Setting, report

_TF_CACHE = {}


class RepNetTFLegacyEngine(Engine):
    id = "repnet_tf_legacy"
    label = "RepNet (TensorFlow, previous set-up)"
    kind = "trained model"
    short = "Your existing TensorFlow RepNet (batch_repnet.py + checkpoint), for comparison."
    about = ("Runs batch_repnet.load_repnet and get_counts with the same arguments as the earlier app. "
             "Use it to confirm that 'RepNet (PyTorch)' matches your existing results, then prefer the PyTorch "
             "version (simpler to install). Untested by the developer.")
    frame_size = 112
    target_fps = 15
    fit = "squash"
    settings = [
        Setting("ckpt_dir", "Checkpoint folder", "text", "", "Folder with the RepNet TensorFlow checkpoint."),
        Setting("code_dir", "Folder containing batch_repnet.py", "text", "", ""),
        Setting("strides", "Strides to try", "text", "1,2,4", "As in the earlier app's standard mode."),
        Setting("within_period_threshold", "Periodicity threshold", "float", 0.28,
                "The earlier app's fixed default.", 0.05, 0.95, 0.01),
        Setting("median_filter", "Median filter", "bool", True, ""),
    ]

    def _paths(self, s=None):
        s = s or {}
        ckpt = Path(s["ckpt_dir"]) if s.get("ckpt_dir") else config.first_existing(config.REPNET_TF_CKPT_CANDIDATES)
        code = Path(s["code_dir"]) if s.get("code_dir") else next(
            (d for d in config.REPNET_TF_CODE_CANDIDATES if (Path(d) / "batch_repnet.py").exists()), None)
        return ckpt, code

    def default_settings(self):
        d = super().default_settings()
        ckpt, code = self._paths()
        d["ckpt_dir"] = str(ckpt) if ckpt else ""
        d["code_dir"] = str(code) if code else ""
        return d

    def check(self, s=None):
        if importlib.util.find_spec("tensorflow") is None and not (config.LEGACY_PKGS_DIR / "tensorflow").exists():
            return False, "TensorFlow is not installed (only needed for this comparison engine)."
        ckpt, code = self._paths(s)
        if not code or not (Path(code) / "batch_repnet.py").exists():
            return False, "batch_repnet.py not found (set its folder in this engine's settings)."
        if not ckpt or not Path(ckpt).exists():
            return False, "RepNet TensorFlow checkpoint folder not found (set it in this engine's settings)."

        # Test whether TensorFlow can be loaded without being blocked by Windows Application Control / Smart App Control
        if "_tf_import_status" not in _TF_CACHE:
            for d in (str(config.LEGACY_PKGS_DIR), str(code)):
                if Path(d).exists() and d not in sys.path:
                    sys.path.insert(0, d)
            try:
                import keras
                import tensorflow.compat.v2 as tf
                _TF_CACHE["_tf_import_status"] = (True, "")
            except Exception as e:
                err_str = str(e)
                if "Application Control" in err_str:
                    msg = "Blocked by Windows Application Control (Smart App Control). Use 'RepNet (PyTorch)' instead."
                else:
                    msg = f"Failed to load TensorFlow ({type(e).__name__}: {e})"
                _TF_CACHE["_tf_import_status"] = (False, msg)

        can_import, import_err = _TF_CACHE["_tf_import_status"]
        if not can_import:
            return False, import_err

        return True, f"Ready (code: {code}; checkpoint: {ckpt})"

    def run(self, clip, settings, progress=None):
        s = {**self.default_settings(), **(settings or {})}
        ok, msg = self.check(s)
        if not ok:
            raise RuntimeError(msg)
        ckpt, code = self._paths(s)
        for d in (str(config.LEGACY_PKGS_DIR), str(code)):
            if Path(d).exists() and d not in sys.path:
                sys.path.insert(0, d)
        report(progress, 0.05, "Loading TensorFlow RepNet (this can take a minute)...")
        try:
            import keras
            import tensorflow.compat.v2 as tf
            tf.keras = keras
            from batch_repnet import load_repnet, get_counts
        except Exception as e:
            if "Application Control" in str(e):
                raise RuntimeError(
                    "TensorFlow DLLs were blocked by Windows Application Control (Smart App Control). "
                    "Please use 'RepNet (PyTorch)' instead, which is simpler and does not require TensorFlow."
                ) from e
            raise
        key = str(ckpt)
        if key not in _TF_CACHE:
            _TF_CACHE.clear()
            _TF_CACHE[key] = load_repnet(key)
        model = _TF_CACHE[key]
        frames, fps, _ = clip.frames(self.target_fps, self.frame_size, self.fit)
        strides = [int(x) for x in str(s["strides"]).replace(" ", "").split(",") if x]
        report(progress, 0.3, "Running RepNet (TensorFlow)...")
        thr = float(s["within_period_threshold"])
        pred_period, pred_score, within_period, per_frame_counts, chosen_stride = get_counts(
            model=model, frames=frames, strides=strides, batch_size=1, threshold=0.20,
            within_period_threshold=thr, constant_speed=False, median_filter=bool(s["median_filter"]),
            fully_periodic=False)
        T = len(frames)
        counts = np.asarray(per_frame_counts, dtype=float).reshape(-1)[:T]
        within = np.asarray(within_period, dtype=float).reshape(-1)[:T]
        if len(counts) < T:
            counts = np.concatenate([counts, np.zeros(T - len(counts))])
            within = np.concatenate([within, np.zeros(T - len(within))])
        report(progress, 1.0, "Done")
        return EngineResult(self.id, fps, np.cumsum(counts), within > thr, score=within,
                            method_used=f"TensorFlow RepNet via batch_repnet.py, stride {chosen_stride}",
                            details={"chosen_stride": int(chosen_stride), "checkpoint": str(ckpt),
                                     "overall_score": float(np.mean(pred_score)) if np.size(pred_score) else None},
                            warnings=["Stride chosen automatically; fix it for the study."] if len(strides) > 1 else [])
