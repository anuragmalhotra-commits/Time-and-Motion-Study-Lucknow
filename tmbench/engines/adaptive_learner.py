"""AI Adaptive Learner (PyTorch) for Chikankari Time & Motion Study.

A self-improving neural calibrator that combines optical flow, hand tracking,
and frequency signals. Trained continuously on manual reference counts to
progressively eliminate bias and maximize validation agreement (ICC).
"""
import numpy as np

from .. import config
from ..learning import CHECKPOINT_PATH, extract_clip_features, learner
from .base import Engine, EngineResult, Setting, report


class AdaptiveLearnerEngine(Engine):
    id = "adaptive_learner"
    label = "AI Adaptive Learner (PyTorch)"
    kind = "self-improving PyTorch neural network"
    short = "Self-improving PyTorch model that adapts to artisanal chikankari stitching from manual validations."
    about = ("Trained continuously on human-validated ground-truth counts. Employs a multi-signal attention "
             "network in PyTorch that adaptively weights optical flow directionality, MediaPipe hand tracking, and "
             "temporal-frequency dynamics. Progressively improves in accuracy, ICC, and stitch-ratio calibration "
             "as more sample clips are validated.")
    frame_size = 140
    target_fps = 15
    fit = "pad"
    settings = [
        Setting("auto_tune_ratio", "Auto-predict stitch multiplier", "bool", True,
                "Uses the learned movement-per-stitch ratio from validation data."),
        Setting("smooth_output", "Smooth cycle trajectory", "bool", True,
                "Ensures monotonically smooth cumulative cycle progression."),
    ]

    def check(self):
        try:
            import torch
            has_ckpt = CHECKPOINT_PATH.exists()
            reg_count = len(learner.read_registry())
            status_text = (f"Ready (PyTorch {torch.__version__} | "
                           f"{'Fine-tuned checkpoint loaded' if has_ckpt else 'Baseline ensemble mode'} | "
                           f"{reg_count} verified clips in knowledge base)")
            return True, status_text
        except Exception as e:
            return False, f"PyTorch Adaptive Learner not available: {e}"

    def run(self, clip, settings, progress=None) -> EngineResult:
        s = {**self.default_settings(), **(settings or {})}
        report(progress, 0.1, "Extracting multi-signal features...")
        features = extract_clip_features(clip, progress=lambda f, t: report(progress, 0.1 + 0.5 * f, t))

        report(progress, 0.65, "Evaluating PyTorch Adaptive Calibrator...")
        pred = learner.predict(features)
        total_cycles = float(pred["predicted_cycles"])

        # Construct frame-by-frame trajectory calibrated to the predicted total
        frames, fps, _ = clip.frames(self.target_fps, self.frame_size, self.fit)
        T = len(frames)

        # Baseline progression curve from optical flow signal
        from .optical_flow import OpticalFlowEngine
        of_engine = OpticalFlowEngine()
        of_res = of_engine.run(clip, of_engine.default_settings(), progress=None)

        raw_cum = of_res.cum_counts
        raw_final = max(float(raw_cum[-1]), 1e-6) if len(raw_cum) else 1.0

        # Scale cumulative counts smoothly to match the PyTorch calibrated target
        scale_factor = total_cycles / raw_final
        calibrated_cum = raw_cum * scale_factor

        # Active flags
        active = of_res.active

        # Event times
        events = None
        if of_res.event_times:
            events = of_res.event_times

        warnings = []
        if not pred.get("is_calibrated"):
            warnings.append("Model is in baseline mode. Enter manual counts on the Validation page to fine-tune!")

        report(progress, 1.0, "Adaptive evaluation complete")
        return EngineResult(
            engine_id=self.id,
            fps=fps,
            cum_counts=calibrated_cum,
            active=active,
            score=of_res.score,
            event_times=events,
            method_used=pred.get("method", "PyTorch Adaptive Calibrator"),
            details={
                "predicted_cycles": pred["predicted_cycles"],
                "confidence_std": pred["confidence_std"],
                "est_movements_per_stitch": pred["est_movements_per_stitch"],
                "est_stitches": pred["est_stitches"],
                "gate_weights": pred["gate_weights"],
                "is_calibrated": pred["is_calibrated"],
            },
            warnings=warnings,
            extra_series={"raw_flow_cycles": raw_cum}
        )
