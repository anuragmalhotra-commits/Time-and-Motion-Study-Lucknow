"""Turns any engine's per-frame output into the same set of time-and-motion indicators,
so engines are compared like for like."""
import numpy as np
import pandas as pd


def _runs(mask):
    """(start, end) index pairs of consecutive True values."""
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    return out


def compute_metrics(cum_counts, active, fps, min_pause_sec=2.0, event_times=None):
    """
    cum_counts: cumulative movement-cycle count per frame (float, non-decreasing)
    active:     True where the engine judged the artisan to be doing repetitive work
    fps:        frame rate of these arrays
    event_times: optional list of cycle times in seconds from the clip start
                 (used for cycle-time statistics when the engine provides them)
    Returns (summary dict, pauses DataFrame, per-fifth DataFrame).
    """
    cum = np.nan_to_num(np.asarray(cum_counts, dtype=float))
    active = np.asarray(active, dtype=bool)
    T = len(cum)
    dur = T / fps if fps > 0 else 0.0
    total = float(cum[-1]) if T else 0.0
    active_sec = active.sum() / fps if fps > 0 else 0.0

    # Cycle times: from engine events if given, else from when the count passes each whole number
    if event_times is not None and len(event_times) > 1:
        ev = np.sort(np.asarray(event_times, dtype=float))
    else:
        t = np.arange(T) / fps
        targets = np.arange(1, int(np.floor(total)) + 1)
        ev = np.interp(targets, cum, t) if len(targets) and np.all(np.diff(cum) >= -1e-9) else np.array([])
    intervals = np.diff(ev) if len(ev) > 1 else np.array([])
    pause_like = 0
    if len(intervals) > 2:
        med = np.median(intervals)
        keep = intervals <= 3.0 * med     # very long gaps are pauses, not cycles
        pause_like = int((~keep).sum())
        intervals = intervals[keep]

    # Pauses: inactive stretches of at least min_pause_sec
    pauses = []
    for a, b in _runs(~active):
        d = (b - a) / fps
        if d >= min_pause_sec:
            pauses.append({"start_sec": round(a / fps, 2), "end_sec": round(b / fps, 2), "duration_sec": round(d, 2)})
    pauses_df = pd.DataFrame(pauses, columns=["start_sec", "end_sec", "duration_sec"])

    # Pace in each fifth of the clip (for change within the task)
    fifths = []
    if T >= 10:
        edges = np.linspace(0, T, 6).astype(int)
        for k in range(5):
            a, b = edges[k], edges[k + 1]
            c = cum[b - 1] - (cum[a - 1] if a > 0 else 0.0)
            act = active[a:b].sum() / fps
            fifths.append({"part": f"{k * 20}-{(k + 1) * 20}%",
                           "cycles": round(float(c), 2),
                           "active_pace_per_min": round(float(c / (act / 60.0)), 2) if act > 0 else np.nan,
                           "active_pct": round(float(active[a:b].mean() * 100.0), 1)})
    fifths_df = pd.DataFrame(fifths)
    slope = np.nan
    if len(fifths_df) == 5 and fifths_df["active_pace_per_min"].notna().sum() >= 3:
        y = fifths_df["active_pace_per_min"].values
        x = np.arange(5)[~np.isnan(y)]
        slope = float(np.polyfit(x, y[~np.isnan(y)], 1)[0])  # change in pace per fifth of the task

    summary = {
        "duration_sec": round(dur, 2),
        "movement_cycles": round(total, 2),
        "overall_pace_per_min": round(total / (dur / 60.0), 2) if dur > 0 else np.nan,
        "active_pace_per_min": round(total / (active_sec / 60.0), 2) if active_sec > 0 else np.nan,
        "active_pct": round(float(active.mean() * 100.0), 1) if T else np.nan,
        "median_cycle_sec": round(float(np.median(intervals)), 3) if len(intervals) else np.nan,
        "mean_cycle_sec": round(float(np.mean(intervals)), 3) if len(intervals) else np.nan,
        "cycle_cv_pct": round(float(np.std(intervals) / np.mean(intervals) * 100.0), 1) if len(intervals) > 2 else np.nan,
        "n_cycles_timed": int(len(intervals)),
        "long_gaps_excluded": pause_like,
        "n_pauses": int(len(pauses_df)),
        "pause_time_sec": round(float(pauses_df["duration_sec"].sum()), 2) if len(pauses_df) else 0.0,
        "pace_change_per_fifth": round(slope, 3) if not np.isnan(slope) else np.nan,
    }
    return summary, pauses_df, fifths_df


def rolling_pace(cum_counts, fps, window_sec=10.0):
    """Cycles per minute over a centred rolling window (for charts)."""
    cum = np.nan_to_num(np.asarray(cum_counts, dtype=float))
    w = max(1, int(round(window_sec * fps)))
    inc = np.diff(np.concatenate([[0.0], cum]))
    return pd.Series(inc).rolling(w, center=True, min_periods=max(1, w // 3)).mean().values * fps * 60.0
