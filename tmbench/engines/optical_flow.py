"""Optical-flow direction cycles (new in version 2).

Measures which way things move between frames (dense optical flow), follows the main
direction of movement over time, and counts one cycle per out-and-back movement.
Because it uses the DIRECTION of movement, not just the amount, it does not have the
half/double-counting problem of the motion-energy method.
"""
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, medfilt

from .base import Engine, EngineResult, Setting, report


class OpticalFlowEngine(Engine):
    id = "optical_flow"
    label = "Optical-flow direction cycles"
    kind = "heuristic"
    short = "Counts out-and-back movements from the direction of motion (no model download needed)."
    about = ("Dense optical flow (OpenCV Farneback) in the cropped area, projected on the main direction of "
             "movement, which is re-estimated every few seconds. One cycle is counted per out-and-back movement. "
             "Any movement in view counts, so crop to the working hand and keep the camera still. "
             "Not yet checked against manual counts on chikankari.")
    frame_size = 160
    target_fps = 15
    settings = [
        Setting("min_cycle_sec", "Shortest cycle to count (s)", "float", 0.30, "Peaks closer than this are merged.", 0.1, 2.0, 0.05),
        Setting("prominence", "Peak prominence (robust z)", "float", 0.6, "Higher = ignore smaller movements.", 0.1, 3.0, 0.1),
        Setting("activity_rel", "Activity threshold (share of strong movement)", "float", 0.15,
                "Frames moving less than this share of the 95th-percentile movement count as pauses.", 0.02, 0.6, 0.01),
        Setting("axis_window_sec", "Direction window (s)", "float", 3.0, "How quickly the main direction may change.", 1.0, 10.0, 0.5),
        Setting("smooth_sec", "Smoothing (s)", "float", 0.10, "Gaussian smoothing of the movement signal.", 0.0, 0.5, 0.02),
    ]

    def run(self, clip, settings, progress=None):
        s = {**self.default_settings(), **(settings or {})}
        report(progress, 0.05, "Reading frames for optical flow...")
        frames, fps, _ = clip.frames(self.target_fps, self.frame_size, self.fit)
        T = len(frames)
        gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        vx, vy, energy, moving_share = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
        for i in range(1, T):
            flow = cv2.calcOpticalFlowFarneback(gray[i - 1], gray[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(flow, axis=2)
            energy[i] = float(mag.mean())
            moving = mag > 0.25
            moving_share[i] = float(moving.mean())
            if moving.sum() > 0.002 * mag.size:
                vx[i] = float(flow[..., 0][moving].mean())
                vy[i] = float(flow[..., 1][moving].mean())
            if i % 100 == 0:
                report(progress, 0.05 + 0.75 * i / T, f"Optical flow {i}/{T} frames")

        # Main direction of movement, re-estimated over a sliding window, kept continuous
        w = max(3, int(round(s["axis_window_sec"] * fps)))
        df = pd.DataFrame({"xx": vx * vx, "yy": vy * vy, "xy": vx * vy})
        r = df.rolling(w, center=True, min_periods=1).mean()
        two_theta = np.unwrap(np.arctan2(2 * r["xy"].values, r["xx"].values - r["yy"].values))
        theta = two_theta / 2.0
        proj = vx * np.cos(theta) + vy * np.sin(theta)
        if s["smooth_sec"] > 0:
            proj = gaussian_filter1d(proj, sigma=max(0.5, s["smooth_sec"] * fps))

        # Activity: enough movement along the main direction
        strength = pd.Series(np.abs(proj)).rolling(max(3, int(fps)), center=True, min_periods=1).mean().values
        ref = np.percentile(strength, 95) if T > 0 else 0.0
        active = strength > s["activity_rel"] * ref if ref > 0 else np.zeros(T, bool)
        if T >= 5:
            active = medfilt(active.astype(float), 5).astype(bool)

        # Count one cycle per positive lobe of the signed movement
        act_vals = proj[active]
        scale = 1.4826 * np.median(np.abs(act_vals - np.median(act_vals))) if act_vals.size else 0.0
        if not scale or scale <= 0:
            scale = np.std(proj) + 1e-9
        z = proj / scale
        dist = max(1, int(round(s["min_cycle_sec"] * fps)))
        pos, _ = find_peaks(z, distance=dist, prominence=s["prominence"])
        neg, _ = find_peaks(-z, distance=dist, prominence=s["prominence"])
        pos = pos[active[pos]] if len(pos) else pos
        neg = neg[active[neg]] if len(neg) else neg

        flags = np.zeros(T)
        flags[pos] = 1.0
        cum = np.cumsum(flags)
        warnings = []
        n_pos, n_neg = len(pos), len(neg)
        if max(n_pos, n_neg) >= 10 and abs(n_pos - n_neg) / max(n_pos, n_neg) > 0.2:
            warnings.append(f"Outward and return movements disagree ({n_pos} vs {n_neg}); the movement may be "
                            "irregular or other things in view are moving. Check the crop.")
        whole_frame = float((moving_share > 0.6).mean()) if T > 1 else 0.0
        if whole_frame > 0.05:
            warnings.append(f"In {whole_frame * 100:.0f}% of frames most of the picture moved - the camera may "
                            "have moved or been bumped. Counts in those parts are unreliable.")
        report(progress, 1.0, "Optical flow done")
        return EngineResult(
            engine_id=self.id, fps=fps, cum_counts=cum, active=active,
            score=np.clip(strength / (ref + 1e-9), 0, 1),
            event_times=(pos / fps).tolist(),
            method_used="Farneback optical flow, direction-tracked, peak counting",
            details={"outward_peaks": n_pos, "return_peaks": n_neg, "frame_size": self.frame_size,
                     "whole_frame_motion_pct": round(whole_frame * 100, 1)},
            warnings=warnings,
            extra_series={"movement_signal_z": z},
        )
