"""Continuous Self-Improving PyTorch Learning Framework for Chikankari Time & Motion Study.

Extracts multi-signal kinematic, frequency (FFT), and temporal features from video motion signals,
and trains an adaptive PyTorch neural calibrator on verified manual counts.
As researchers validate sample clips, the model continuously updates its weights, learns
artisan-specific and motif-specific dynamics, and progressively maximizes accuracy and ICC.
"""
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from . import config

logger = logging.getLogger(__name__)

LEARNING_DIR = config.RESULTS_DIR / "learning"
REGISTRY_FILE = LEARNING_DIR / "ground_truth_registry.json"
CHECKPOINT_PATH = config.MODELS_DIR / "adaptive_pytorch_calibrator.pt"
HISTORY_FILE = LEARNING_DIR / "learning_history.json"


@dataclass
class ClipFeatures:
    duration_sec: float
    fps: float
    # Optical flow dynamics
    of_raw_count: float
    of_pace_per_min: float
    of_active_ratio: float
    of_energy_mean: float
    of_energy_std: float
    of_energy_95p: float
    of_fft_peak_hz: float
    of_fft_power_ratio: float
    of_symmetry_ratio: float
    # Hand landmark signals
    hl_raw_count: float
    hl_detection_rate: float
    hl_wrist_disp_mean: float
    hl_finger_speed_mean: float
    # Motion energy
    me_raw_count: float
    # RepNet
    repnet_raw_count: float
    repnet_confidence: float
    # Movement rhythmicity & pauses
    cycle_cv_pct: float
    n_pauses: int
    pause_time_sec: float
    # Context
    is_with_glasses: float
    is_motif_a: float
    is_motif_b: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.duration_sec, self.fps,
            self.of_raw_count, self.of_pace_per_min, self.of_active_ratio,
            self.of_energy_mean, self.of_energy_std, self.of_energy_95p,
            self.of_fft_peak_hz, self.of_fft_power_ratio, self.of_symmetry_ratio,
            self.hl_raw_count, self.hl_detection_rate, self.hl_wrist_disp_mean, self.hl_finger_speed_mean,
            self.me_raw_count, self.repnet_raw_count, self.repnet_confidence,
            self.cycle_cv_pct, float(self.n_pauses), self.pause_time_sec,
            self.is_with_glasses, self.is_motif_a, self.is_motif_b
        ], dtype=np.float32)

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
            "duration_sec", "fps",
            "of_raw_count", "of_pace_per_min", "of_active_ratio",
            "of_energy_mean", "of_energy_std", "of_energy_95p",
            "of_fft_peak_hz", "of_fft_power_ratio", "of_symmetry_ratio",
            "hl_raw_count", "hl_detection_rate", "hl_wrist_disp_mean", "hl_finger_speed_mean",
            "me_raw_count", "repnet_raw_count", "repnet_confidence",
            "cycle_cv_pct", "n_pauses", "pause_time_sec",
            "is_with_glasses", "is_motif_a", "is_motif_b"
        ]


def extract_clip_features(clip, meta: Optional[dict] = None, base_outcomes: Optional[list] = None,
                          progress=None) -> ClipFeatures:
    """Extracts compact 24-dimensional motion & frequency feature vector from a clip without saving video."""
    meta = meta or {}
    duration = max(0.5, float(clip.duration))

    # Context flags
    cond_str = str(meta.get("condition", "")).lower()
    is_glasses = 1.0 if "with" in cond_str and "without" not in cond_str else (
        -1.0 if "without" in cond_str else 0.0)
    motif_str = str(meta.get("motif", "")).upper()
    is_motif_a = 1.0 if motif_str == "A" else 0.0
    is_motif_b = 1.0 if motif_str == "B" else 0.0

    # Base engine counts if already available in outcomes
    outcomes_dict = {}
    if base_outcomes:
        for o in base_outcomes:
            if o.get("ok"):
                outcomes_dict[o["engine_id"]] = o

    # Extract optical flow & signal dynamics
    if progress:
        progress(0.1, "Extracting kinematic features...")
    frames, fps, _ = clip.frames(15, 140, fit="pad")
    T = len(frames)
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

    vx, vy, energy = np.zeros(T), np.zeros(T), np.zeros(T)
    for i in range(1, T):
        flow = cv2.calcOpticalFlowFarneback(gray[i - 1], gray[i], None, 0.5, 3, 12, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        energy[i] = float(mag.mean())
        moving = mag > 0.25
        if moving.sum() > 0.002 * mag.size:
            vx[i] = float(flow[..., 0][moving].mean())
            vy[i] = float(flow[..., 1][moving].mean())

    # Movement axis projection
    w = max(3, int(round(3.0 * fps)))
    df_dir = pd.DataFrame({"xx": vx * vx, "yy": vy * vy, "xy": vx * vy})
    r = df_dir.rolling(w, center=True, min_periods=1).mean()
    two_theta = np.unwrap(np.arctan2(2 * r["xy"].values, r["xx"].values - r["yy"].values))
    theta = two_theta / 2.0
    proj = vx * np.cos(theta) + vy * np.sin(theta)
    proj_smooth = gaussian_filter1d(proj, sigma=max(0.5, 0.1 * fps))

    # Frequency analysis via FFT
    sig = proj_smooth - np.mean(proj_smooth)
    fft_vals = np.abs(np.fft.rfft(sig))
    fft_freqs = np.fft.rfftfreq(len(sig), d=1.0 / fps)
    valid_freq = (fft_freqs >= 0.2) & (fft_freqs <= 3.5)
    if valid_freq.any():
        peak_idx = np.argmax(fft_vals[valid_freq])
        fft_peak_hz = float(fft_freqs[valid_freq][peak_idx])
        tot_power = np.sum(fft_vals[valid_freq] ** 2) + 1e-9
        peak_power = float(fft_vals[valid_freq][peak_idx] ** 2)
        fft_power_ratio = float(peak_power / tot_power)
    else:
        fft_peak_hz = 0.8
        fft_power_ratio = 0.0

    # Peaks & symmetry
    scale = 1.4826 * np.median(np.abs(proj_smooth - np.median(proj_smooth))) + 1e-6
    z = proj_smooth / scale
    dist = max(1, int(round(0.3 * fps)))
    pos_peaks, _ = find_peaks(z, distance=dist, prominence=0.5)
    neg_peaks, _ = find_peaks(-z, distance=dist, prominence=0.5)
    n_pos, n_neg = len(pos_peaks), len(neg_peaks)
    symmetry = float(min(n_pos, n_neg) / max(max(n_pos, n_neg), 1))

    # Activity & pauses
    strength = pd.Series(np.abs(proj_smooth)).rolling(max(3, int(fps)), center=True, min_periods=1).mean().values
    ref_str = np.percentile(strength, 95) if T > 0 else 1.0
    act = strength > (0.15 * ref_str)
    act_ratio = float(np.mean(act)) if T > 0 else 0.0

    # Pause intervals
    pause_runs = []
    in_pause, p_len = False, 0
    for a in act:
        if not a:
            p_len += 1
            in_pause = True
        else:
            if in_pause and p_len >= int(2.0 * fps):
                pause_runs.append(p_len / fps)
            in_pause, p_len = False, 0
    if in_pause and p_len >= int(2.0 * fps):
        pause_runs.append(p_len / fps)

    n_pauses = len(pause_runs)
    pause_time = float(sum(pause_runs))

    # Cycle CV
    if len(pos_peaks) >= 3:
        diffs = np.diff(pos_peaks) / fps
        cycle_cv = float(np.std(diffs) / max(np.mean(diffs), 1e-6) * 100.0)
    else:
        cycle_cv = 20.0

    # Populate engine values
    of_cnt = float(outcomes_dict.get("optical_flow", {}).get("summary", {}).get("movement_cycles", n_pos))
    hl_cnt = float(outcomes_dict.get("hand_landmarks", {}).get("summary", {}).get("movement_cycles", 0.0))
    me_cnt = float(outcomes_dict.get("motion_energy", {}).get("summary", {}).get("movement_cycles", 0.0))
    rep_cnt = float(outcomes_dict.get("repnet_torch", {}).get("summary", {}).get(
        "movement_cycles", outcomes_dict.get("repnet_tf_legacy", {}).get("summary", {}).get("movement_cycles", 0.0)))
    rep_conf = float(outcomes_dict.get("repnet_torch", {}).get("result", {}).score.mean() if (
        "repnet_torch" in outcomes_dict and outcomes_dict["repnet_torch"].get("result") and
        outcomes_dict["repnet_torch"]["result"].score is not None) else 0.5)

    # If hand landmarks were not pre-calculated, run a quick landmark check if model exists
    hl_detect_rate = 0.0
    if hl_cnt <= 0:
        try:
            from .engines.hand_landmarks import HandLandmarkEngine
            hle = HandLandmarkEngine()
            if hle.check()[0] and T <= 450:  # within 30s
                hl_res = hle.run(clip, hle.default_settings())
                hl_cnt = float(hl_res.cum_counts[-1]) if len(hl_res.cum_counts) else 0.0
                hl_detect_rate = float(np.mean(hl_res.active)) if len(hl_res.active) else 0.0
        except Exception:
            pass

    return ClipFeatures(
        duration_sec=round(duration, 2),
        fps=round(float(fps), 1),
        of_raw_count=round(of_cnt, 2),
        of_pace_per_min=round(of_cnt / (duration / 60.0), 2) if duration > 0 else 0.0,
        of_active_ratio=round(act_ratio, 3),
        of_energy_mean=round(float(np.mean(energy)), 3),
        of_energy_std=round(float(np.std(energy)), 3),
        of_energy_95p=round(float(np.percentile(energy, 95)), 3),
        of_fft_peak_hz=round(fft_peak_hz, 3),
        of_fft_power_ratio=round(fft_power_ratio, 3),
        of_symmetry_ratio=round(symmetry, 3),
        hl_raw_count=round(hl_cnt, 2),
        hl_detection_rate=round(hl_detect_rate or float(outcomes_dict.get("hand_landmarks", {}).get("result", {}).active.mean()
                                      if "hand_landmarks" in outcomes_dict and outcomes_dict["hand_landmarks"].get("result")
                                      else 0.0), 3),
        hl_wrist_disp_mean=0.0,
        hl_finger_speed_mean=0.0,
        me_raw_count=round(me_cnt, 2),
        repnet_raw_count=round(rep_cnt, 2),
        repnet_confidence=round(rep_conf, 3),
        cycle_cv_pct=round(cycle_cv, 1),
        n_pauses=n_pauses,
        pause_time_sec=round(pause_time, 2),
        is_with_glasses=is_glasses,
        is_motif_a=is_motif_a,
        is_motif_b=is_motif_b,
    )


# ----------------------------------------------------------------------------
# PyTorch Adaptive Model
# ----------------------------------------------------------------------------
def _get_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


class ChikankariCycleNet:
    """Adaptive PyTorch Neural Network with Attention Gating and Multiplicative Ratio Calibration."""
    @staticmethod
    def create_model(in_features: int = 24, hidden_dim: int = 64):
        torch, nn, F = _get_torch()

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.BatchNorm1d(in_features)
                self.backbone = nn.Sequential(
                    nn.Linear(in_features, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.12),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.08),
                )
                # Gating weights over candidate signals: [Optical Flow, Hand Landmarks, RepNet/Heuristic]
                self.gate = nn.Sequential(
                    nn.Linear(hidden_dim, 3),
                    nn.Softmax(dim=-1)
                )
                # Multiplicative scale head: learns ratio between raw hand cycles and completed stitches
                self.scale_head = nn.Sequential(
                    nn.Linear(hidden_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, 1)
                )
                # Fine residual adjustment
                self.residual = nn.Sequential(
                    nn.Linear(hidden_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, 1)
                )
                # Uncertainty estimator (confidence std)
                self.uncertainty = nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                    nn.Softplus()
                )

            def forward(self, x, base_candidates):
                # x: [B, in_features]
                # base_candidates: [B, 3] (of_count, hl_count, repnet_or_me_count)
                if x.shape[0] > 1:
                    x_norm = self.norm(x)
                else:
                    x_norm = x
                feat = self.backbone(x_norm)
                gates = self.gate(feat)  # [B, 3]
                base_blend = (gates * base_candidates).sum(dim=-1, keepdim=True)  # [B, 1]

                # Softplus scaling factor (e.g. 0.5 for a 2:1 movements-per-stitch ratio, 1.0 for 1:1)
                scale = F.softplus(self.scale_head(feat) + 0.54)  # softplus(0.54) ≈ 1.0 initial
                scale = torch.clamp(scale, min=0.05, max=4.0)

                res = self.residual(feat)
                cycle_pred = torch.clamp(base_blend * scale + res, min=0.0)

                # Learned movements-per-stitch ratio = 1 / scale
                ratio_pred = torch.clamp(1.0 / torch.clamp(scale, min=0.1, max=5.0), min=0.5, max=5.0)
                unc = self.uncertainty(feat) + 0.1
                return cycle_pred, ratio_pred, unc, gates

        return _Model()


class ContinuousLearner:
    """Manages ground-truth sample registry, training iterations, and inference."""

    def __init__(self):
        LEARNING_DIR.mkdir(parents=True, exist_ok=True)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model = None
        self._load_or_init()

    def _load_or_init(self):
        try:
            torch, _, _ = _get_torch()
            self.model = ChikankariCycleNet.create_model()
            if CHECKPOINT_PATH.exists():
                ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
                self.model.load_state_dict(ckpt["state_dict"], strict=False)
                self.model.eval()
                logger.info("Loaded trained PyTorch calibrator checkpoint.")
        except Exception as e:
            logger.warning(f"Could not initialize PyTorch model: {e}")
            self.model = None

    def read_registry(self) -> List[dict]:
        if not REGISTRY_FILE.exists():
            return []
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
                # Filter out corrupted records with NaN, duration < 5s, or unnatural pace
                clean = []
                for r in records:
                    f = r.get("features", {})
                    dur = float(f.get("duration_sec", 0.0) or 0.0)
                    ref = float(r.get("reference_count", 0.0) or 0.0)
                    pace = ref / (dur / 60.0) if dur > 0 else 0
                    if dur >= 5.0 and ref > 0 and 2.0 <= pace <= 120.0 and not np.isnan(dur) and not np.isnan(ref):
                        clean.append(r)
                return clean
        except Exception:
            return []

    def write_registry(self, records: List[dict]):
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def add_ground_truth(self, clip_label: str, video_file: str, start_sec: float, end_sec: float,
                         reference_count: float, features: ClipFeatures, meta: Optional[dict] = None) -> int:
        records = self.read_registry()
        found = False
        new_rec = {
            "clip_label": str(clip_label),
            "video_file": str(video_file),
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "reference_count": float(reference_count),
            "features": asdict(features),
            "meta": meta or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        for i, r in enumerate(records):
            if (r.get("video_file") == video_file and
                    abs(float(r.get("start_sec", 0)) - float(start_sec)) < 0.2 and
                    abs(float(r.get("end_sec", 0)) - float(end_sec)) < 0.2):
                records[i] = new_rec
                found = True
                break
        if not found:
            records.append(new_rec)
        self.write_registry(records)
        return len(records)

    def train(self, epochs: int = 150, lr: float = 0.003, progress_callback=None) -> dict:
        """Trains or fast-calibrates the PyTorch model on all validated clips."""
        torch, nn, F = _get_torch()
        records = self.read_registry()
        if len(records) == 0:
            return {"ok": False, "error": "No valid clips in knowledge base. Please enter manual counts first."}

        # Prepare dataset
        X, Y, C = [], [], []
        for r in records:
            f = ClipFeatures(**r["features"])
            arr = f.to_array()
            ref = float(r["reference_count"])
            c_of = f.of_raw_count
            c_hl = f.hl_raw_count if f.hl_raw_count > 0 else c_of
            c_rep = f.repnet_raw_count if f.repnet_raw_count > 0 else f.me_raw_count
            if c_rep <= 0:
                c_rep = c_of
            X.append(arr)
            Y.append(ref)
            C.append([c_of, c_hl, c_rep])

        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        Y_t = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(1)
        C_t = torch.tensor(np.array(C), dtype=torch.float32)

        model = ChikankariCycleNet.create_model()

        # If few samples (1 or 2 clips), apply fast empirical ratio calibration
        if len(records) < 3:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            loss_fn = nn.HuberLoss(delta=1.0)
            history_losses = []
            for ep in range(epochs):
                optimizer.zero_grad()
                preds, _, unc, _ = model(X_t, C_t)
                loss = loss_fn(preds, Y_t) + 0.05 * torch.mean(unc)
                loss.backward()
                optimizer.step()
                history_losses.append(float(loss.item()))
                if progress_callback and (ep + 1) % 20 == 0:
                    progress_callback((ep + 1) / epochs, f"Calibrating ratio (Loss: {loss.item():.3f})")
        else:
            # Full neural optimization with AdamW and cosine annealing
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
            loss_fn = nn.HuberLoss(delta=1.5)

            history_losses = []
            for ep in range(epochs):
                optimizer.zero_grad()
                preds, _, unc, _ = model(X_t, C_t)
                loss = loss_fn(preds, Y_t) + 0.05 * torch.mean(unc)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                history_losses.append(float(loss.item()))
                if progress_callback and (ep + 1) % 15 == 0:
                    progress_callback((ep + 1) / epochs, f"Training epoch {ep + 1}/{epochs} (Loss: {loss.item():.3f})")

        model.eval()
        with torch.no_grad():
            final_preds, final_ratios, final_unc, final_gates = model(X_t, C_t)
            preds_np = final_preds.squeeze().cpu().numpy()
            y_np = Y_t.squeeze().cpu().numpy()
            if np.ndim(preds_np) == 0:
                preds_np = np.array([float(preds_np)])
                y_np = np.array([float(y_np)])
            learned_ratio_val = float(final_ratios.mean().item())

        diff = preds_np - y_np
        pct_err = np.abs(diff) / np.maximum(y_np, 1e-6) * 100.0
        mae = float(np.mean(np.abs(diff)))
        nmae = float(np.mean(pct_err))

        # Checkpoint
        state = {
            "state_dict": model.state_dict(),
            "records_count": len(records),
            "mae": mae,
            "nmae": nmae,
            "learned_ratio": learned_ratio_val,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save(state, CHECKPOINT_PATH)
        self.model = model

        # Update history
        hist = self.get_history()
        hist.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": len(records),
            "mae": round(mae, 2),
            "nmae_pct": round(nmae, 1),
            "learned_ratio": round(learned_ratio_val, 2),
            "loss_curve": history_losses[::max(1, len(history_losses) // 30)]
        })
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2)

        return {
            "ok": True,
            "n_samples": len(records),
            "mae": round(mae, 2),
            "nmae_pct": round(nmae, 1),
            "learned_ratio": round(learned_ratio_val, 2),
            "loss": history_losses[-1],
            "checkpoint": str(CHECKPOINT_PATH)
        }

    def predict(self, features: ClipFeatures) -> dict:
        """Evaluates the model on clip features, with intelligent fallback if uninitialized."""
        c_of = features.of_raw_count
        c_hl = features.hl_raw_count if features.hl_raw_count > 0 else c_of
        c_rep = features.repnet_raw_count if features.repnet_raw_count > 0 else features.me_raw_count
        if c_rep <= 0:
            c_rep = c_of

        # If model is loaded from checkpoint, run neural inference
        if self.model is not None and CHECKPOINT_PATH.exists():
            try:
                torch, _, _ = _get_torch()
                self.model.eval()
                with torch.no_grad():
                    x_t = torch.tensor(features.to_array(), dtype=torch.float32).unsqueeze(0)
                    c_t = torch.tensor([[c_of, c_hl, c_rep]], dtype=torch.float32)
                    pred, ratio, unc, gates = self.model(x_t, c_t)
                    count_val = float(pred.item())
                    ratio_val = float(ratio.item())
                    unc_val = float(unc.item())
                    gate_vals = gates.squeeze().cpu().numpy().tolist()
                return {
                    "method": "Adaptive Neural Calibrator (PyTorch fine-tuned)",
                    "predicted_cycles": round(count_val, 2),
                    "confidence_std": round(unc_val, 2),
                    "est_movements_per_stitch": round(ratio_val, 2),
                    "est_stitches": round(count_val, 1),
                    "gate_weights": {
                        "optical_flow": round(gate_vals[0], 2),
                        "hand_landmarks": round(gate_vals[1], 2),
                        "repnet": round(gate_vals[2], 2)
                    },
                    "is_calibrated": True
                }
            except Exception as e:
                logger.warning(f"Inference error in learned model: {e}")

        # Baseline ensemble mode (before manual validation fine-tuning)
        w_hl = 0.5 if features.hl_detection_rate > 0.4 else 0.1
        w_of = 0.4
        w_rep = 0.3 if features.repnet_confidence > 0.4 else 0.1
        tot = w_hl + w_of + w_rep
        blend = (w_of * c_of + w_hl * c_hl + w_rep * c_rep) / tot
        return {
            "method": "Ensemble (Pre-calibration baseline)",
            "predicted_cycles": round(blend, 2),
            "confidence_std": round(0.12 * blend, 2),
            "est_movements_per_stitch": 1.0,
            "est_stitches": round(blend, 1),
            "gate_weights": {
                "optical_flow": round(w_of / tot, 2),
                "hand_landmarks": round(w_hl / tot, 2),
                "repnet": round(w_rep / tot, 2)
            },
            "is_calibrated": False
        }

    def get_history(self) -> List[dict]:
        if not HISTORY_FILE.exists():
            return []
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []


# Global continuous learner instance
learner = ContinuousLearner()
