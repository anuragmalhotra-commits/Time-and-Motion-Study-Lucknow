"""Hand-landmark cycles (Google MediaPipe HandLandmarker).

Follows one hand's index fingertip, removes slow drift, projects on the main direction
of movement and counts peaks. Only counts where the hand is actually seen.
"""
import importlib.util

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .. import config
from .base import Engine, EngineResult, Setting, report

WRIST, INDEX_TIP = 0, 8


def _runs(mask, min_len):
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


class HandLandmarkEngine(Engine):
    id = "hand_landmarks"
    label = "Hand-landmark cycles (MediaPipe)"
    kind = "trained model (hand detector) + rules"
    short = "Tracks the fingertip of one hand and counts its out-and-back movements."
    about = ("Google MediaPipe HandLandmarker finds 21 points per hand in each frame. The index fingertip of one "
             "hand (the one seen most and moving most, or the one you choose) is followed; slow drift is removed "
             "and peaks along the main direction are counted. Frames where the hand is hidden are not counted. "
             "Hand labels (Left/Right) may be mirrored for rear-camera video. Not yet checked against manual "
             "counts on chikankari.")
    frame_size = 384
    target_fps = 15
    settings = [
        Setting("hand", "Hand to follow", "choice", "auto", "auto = seen most and moving most.", options=["auto", "Left", "Right"]),
        Setting("min_cycle_sec", "Shortest cycle to count (s)", "float", 0.40, "", 0.1, 2.0, 0.05),
        Setting("prominence", "Peak prominence (robust z)", "float", 0.5, "", 0.1, 3.0, 0.1),
        Setting("max_gap_sec", "Fill gaps up to (s)", "float", 1.0, "Short losses of the hand are bridged.", 0.0, 3.0, 0.25),
        Setting("drift_window_sec", "Drift removal window (s)", "float", 3.0, "", 1.0, 10.0, 0.5),
        Setting("min_confidence", "Detection confidence", "float", 0.3, "", 0.05, 0.9, 0.05),
    ]

    def model_path(self):
        p = config.first_existing(config.HAND_MODEL_CANDIDATES)
        return str(p) if p else None

    def check(self):
        if importlib.util.find_spec("mediapipe") is None:
            return False, "The mediapipe package is not installed (run Setup_Environment.bat)."
        if not self.model_path():
            return False, f"hand_landmarker.task not found in {config.MODELS_DIR} (Setup_Environment.bat downloads it)."
        return True, f"Ready (model: {self.model_path()})"

    def run(self, clip, settings, progress=None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        s = {**self.default_settings(), **(settings or {})}
        ok, msg = self.check()
        if not ok:
            raise RuntimeError(msg)
        report(progress, 0.05, f"Reading frames at {self.frame_size}px...")
        frames, fps, _ = clip.frames(self.target_fps, self.frame_size, self.fit)
        T = len(frames)
        tracks = {lab: {k: np.full(T, np.nan) for k in ("ix", "iy", "wy")} | {"score": np.zeros(T)}
                  for lab in ("Left", "Right")}
        any_hand = np.zeros(T, bool)
        errors = 0
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.model_path()),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=s["min_confidence"], min_hand_presence_confidence=s["min_confidence"],
            min_tracking_confidence=s["min_confidence"])
        det = vision.HandLandmarker.create_from_options(opts)
        try:
            for i, f in enumerate(frames):
                try:
                    res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)),
                                               int(round(i * 1000.0 / fps)))
                except Exception:
                    errors += 1
                    continue
                if res and res.hand_landmarks:
                    any_hand[i] = True
                    for k, lms in enumerate(res.hand_landmarks):
                        try:
                            cat = res.handedness[k][0]
                            lab, sc = cat.category_name, float(cat.score)
                        except Exception:
                            lab, sc = "Right", 0.0
                        tr = tracks.get(lab)
                        if tr is None or sc <= tr["score"][i]:
                            continue
                        tr["score"][i] = sc
                        tr["ix"][i], tr["iy"][i], tr["wy"][i] = lms[INDEX_TIP].x, lms[INDEX_TIP].y, lms[WRIST].y
                if i % 50 == 0:
                    report(progress, 0.1 + 0.8 * i / T, f"Tracking hands {i}/{T}")
        finally:
            try:
                det.close()
            except Exception:
                pass

        gap = max(1, int(round(fps * s["max_gap_sec"])))
        drift = max(3, int(round(fps * s["drift_window_sec"])) | 1)
        min_run = max(5, int(2 * fps))

        def signal(tr):
            ix = pd.Series(tr["ix"]).interpolate(limit=gap, limit_area="inside")
            iy = pd.Series(tr["iy"]).interpolate(limit=gap, limit_area="inside")
            valid = (~ix.isna() & ~iy.isna()).values
            sig = np.full(T, np.nan)
            for a, b in _runs(valid, min_run):
                xy = np.column_stack([ix.values[a:b], iy.values[a:b]])
                d = xy - pd.DataFrame(xy).rolling(drift, center=True, min_periods=1).median().values
                d = d - d.mean(axis=0)
                try:
                    _, _, vt = np.linalg.svd(d, full_matrices=False)
                    sig[a:b] = d @ vt[0]
                except np.linalg.LinAlgError:
                    sig[a:b] = d[:, 1]
            return sig

        cands = {}
        for lab, tr in tracks.items():
            sg = signal(tr)
            seen = int(np.sum(~np.isnan(tr["ix"])))
            amp = float(np.nanstd(sg)) if np.any(~np.isnan(sg)) else 0.0
            cands[lab] = (sg, seen, amp)
        primary = s["hand"] if s["hand"] in cands else max(cands, key=lambda k: cands[k][1] * cands[k][2])
        sig = cands[primary][0]
        analysable = ~np.isnan(sig)
        z = np.full(T, np.nan)
        if analysable.any():
            v = sig[analysable]
            med = np.median(v)
            mad = 1.4826 * np.median(np.abs(v - med)) + 1e-9
            z[analysable] = (v - med) / mad
        flags = np.zeros(T)
        dist = max(1, int(round(s["min_cycle_sec"] * fps)))
        n_pos = n_neg = 0
        for a, b in _runs(analysable, min_run):
            pk, _ = find_peaks(z[a:b], distance=dist, prominence=s["prominence"])
            vl, _ = find_peaks(-z[a:b], distance=dist, prominence=s["prominence"])
            flags[a + pk] = 1.0
            n_pos += len(pk)
            n_neg += len(vl)
        # Active = hand analysable AND moving (a visible but still hand is a pause)
        strength = pd.Series(np.abs(np.nan_to_num(z))).rolling(max(3, int(fps)), center=True, min_periods=1).mean().values
        ref = np.percentile(strength[analysable], 95) if analysable.any() else 0.0
        active = analysable & (strength > 0.15 * ref) if ref > 0 else analysable.copy()
        flags[~active] = 0.0
        cum = np.cumsum(flags)
        events = (np.where(flags > 0)[0] / fps).tolist()

        warnings = []
        seen_pct = float(any_hand.mean() * 100) if T else 0.0
        anal_pct = float(analysable.mean() * 100) if T else 0.0
        if anal_pct < 70:
            warnings.append(f"The chosen hand could be analysed in only {anal_pct:.0f}% of frames; "
                            "counts cover only those parts.")
        if errors:
            warnings.append(f"{errors} frames could not be processed.")
        if max(n_pos, n_neg) >= 10 and abs(n_pos - n_neg) / max(n_pos, n_neg) > 0.2:
            warnings.append(f"Peaks and troughs disagree ({n_pos} vs {n_neg}); the fingertip path is irregular.")
        report(progress, 1.0, "Hand tracking done")
        return EngineResult(
            self.id, fps, cum, active, score=None, event_times=events,
            method_used=f"MediaPipe HandLandmarker at {self.frame_size}px, following the {primary} hand",
            details={"hand_seen_pct": round(seen_pct, 1), "analysable_pct": round(anal_pct, 1),
                     "primary_hand": primary, "peaks": n_pos, "troughs": n_neg},
            warnings=warnings,
            extra_series={"any_hand_detected": any_hand, "fingertip_signal_z": z},
        )
