"""Motion-energy autocorrelation - the counting method of the earlier app's
"TransRAC-style" fast mode, kept for comparison.

Known limits from synthetic tests: correct only for cycles of roughly 0.8-1.6 s.
Faster cycles are under-counted (a warning is given); slow, smooth back-and-forth
cycles can be counted twice (this cannot be detected automatically).
"""
import cv2
import numpy as np
from scipy.signal import find_peaks, medfilt

from .base import Engine, EngineResult, Setting, report


def motion_energy_density(frames_gray, fps, scales=(1, 2, 4), window_size=None, active_thresh=0.28,
                          min_period_sec=0.80, max_period_sec=3.50):
    T = len(frames_gray)
    if T < 4:
        return np.zeros(T), np.zeros(T), 0.0
    if window_size is None:
        window_size = max(45, int(fps * 4.5))
    energies = []
    for s in scales:
        diffs = [float(np.mean(cv2.absdiff(frames_gray[min(T - 1, i + s)], frames_gray[max(0, i - s)])))
                 for i in range(T)]
        energies.append(np.array(diffs, dtype=np.float32))
    motion = np.mean(energies, axis=0)
    lo, hi = motion.min(), motion.max()
    norm = (motion - lo) / (hi - lo + 1e-6) if hi > lo else np.zeros(T, np.float32)

    density = np.zeros(T, np.float32)
    period = np.zeros(T, np.float32)
    fast = np.zeros(T, bool)
    half = window_size // 2
    min_lag = max(4, int(fps * min_period_sec))
    max_lag = min(window_size - 1, int(fps * max_period_sec))
    for i in range(T):
        w = norm[max(0, i - half):min(T, i + half)]
        if len(w) >= max_lag and max_lag > min_lag and np.std(w) > 0.02:
            sig = w - w.mean()
            ac_full = np.correlate(sig, sig, mode="full")
            mid = len(sig) - 1
            ac = ac_full[mid: mid + max_lag + 1]
            if ac[0] > 0:
                ac = ac / ac[0]
            region = ac[min_lag: max_lag + 1]
            peaks, props = find_peaks(region, height=0.10, prominence=0.04)
            if len(peaks):
                b = peaks[np.argmax(props["peak_heights"])]
                lag, val = b + min_lag, region[b]
                if val >= active_thresh:
                    density[i] = 1.0 / lag
                    period[i] = float(np.clip(val, 0, 1))
                    sub, _ = find_peaks(ac[2:min_lag], height=max(0.3, 0.6 * val))
                    if len(sub) and lag / float(sub[0] + 2) > 2.5:
                        fast[i] = True
                else:
                    period[i] = float(max(0.0, val))
    if T >= 7:
        period = medfilt(period, 5)
        k = min(T, max(3, int(fps)))
        k = k - 1 if k % 2 == 0 else k
        density = np.convolve(density, np.ones(k) / k, mode="same")[:T]
    counted = float((density > 0).sum())
    fast_pct = float(fast.sum()) / counted * 100.0 if counted else 0.0
    return density, period, fast_pct


class MotionEnergyEngine(Engine):
    id = "motion_energy"
    label = "Motion-energy autocorrelation (previous app)"
    kind = "heuristic"
    short = "The earlier app's fast pixel method - kept for comparison only."
    about = ("Measures how much the picture changes between frames and looks for a repeating rhythm. "
             "This was the earlier app's 'TransRAC-style' fast mode; it is not TransRAC. In synthetic tests "
             "it was right only for cycles of about 0.8-1.6 s: faster work was under-counted (flagged) and slow, "
             "smooth cycles were counted twice (not detectable automatically).")
    frame_size = 112
    target_fps = 15
    settings = [
        Setting("active_thresh", "Active threshold", "float", 0.28, "Periodicity needed to count a frame as active.", 0.05, 0.9, 0.01),
        Setting("window_sec", "Analysis window (s)", "float", 4.5, "Length of the sliding window.", 2.0, 10.0, 0.5),
    ]

    def run(self, clip, settings, progress=None):
        s = {**self.default_settings(), **(settings or {})}
        report(progress, 0.1, "Reading frames...")
        frames, fps, _ = clip.frames(self.target_fps, self.frame_size, self.fit)
        gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        report(progress, 0.4, "Motion-energy autocorrelation...")
        density, period, fast_pct = motion_energy_density(gray, fps, window_size=int(fps * s["window_sec"]),
                                                          active_thresh=s["active_thresh"])
        active = period >= s["active_thresh"]
        cum = np.cumsum(density * active)
        warnings = []
        if fast_pct > 20:
            warnings.append(f"In {fast_pct:.0f}% of counted frames the movement looks faster than one cycle per "
                            "0.8 s, which this method cannot count: the count is probably too low.")
        report(progress, 1.0, "Done")
        return EngineResult(self.id, fps, cum, active, score=period,
                            method_used="motion-energy autocorrelation (pixel differencing)",
                            details={"fast_cadence_flag_pct": round(fast_pct, 1)}, warnings=warnings)
