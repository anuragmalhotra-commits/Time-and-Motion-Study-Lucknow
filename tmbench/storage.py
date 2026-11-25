"""Saving results: one folder per analysis run, plus one row per engine in
results/master_results.csv (the file to use for the crossover analysis)."""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, config

META_FIELDS = ["artisan_id", "visit", "motif", "condition", "observer", "observer_minutes", "notes"]


def _safe(text, fallback="NA"):
    t = re.sub(r"[^A-Za-z0-9_-]+", "-", str(text or "")).strip("-")
    return t[:40] or fallback


def make_run_dir(meta, video_name):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = "_".join([stamp, _safe(meta.get("artisan_id"), "UNLABELLED"), _safe(meta.get("visit")),
                     _safe(meta.get("motif")), _safe(meta.get("condition")), _safe(Path(video_name).stem)[:20]])
    d = config.RESULTS_DIR / "runs" / name
    n = 1
    while d.exists():
        n += 1
        d = config.RESULTS_DIR / "runs" / f"{name}_{n}"
    d.mkdir(parents=True)
    return d


def save_engine_output(run_dir, outcome):
    """Writes frame-level, pauses, per-fifth and summary files for one engine."""
    eid = outcome["engine_id"]
    if not outcome.get("ok"):
        (run_dir / f"{eid}_ERROR.txt").write_text(outcome.get("error", "unknown error"), encoding="utf-8")
        return
    r = outcome["result"]
    T = len(r.cum_counts)
    cols = {"time_sec": np.round(np.arange(T) / r.fps, 3), "cumulative_cycles": np.round(r.cum_counts, 4),
            "active": np.asarray(r.active, bool)}
    if r.score is not None:
        cols["score"] = np.round(np.asarray(r.score, float), 4)
    for k, v in (r.extra_series or {}).items():
        if len(v) == T:
            cols[k] = v
    pd.DataFrame(cols).to_csv(run_dir / f"{eid}_frames.csv", index=False, encoding="utf-8-sig")
    outcome["pauses"].to_csv(run_dir / f"{eid}_pauses.csv", index=False, encoding="utf-8-sig")
    outcome["fifths"].to_csv(run_dir / f"{eid}_by_fifth.csv", index=False, encoding="utf-8-sig")
    if r.event_times:
        pd.DataFrame({"cycle_time_sec": np.round(r.event_times, 3)}).to_csv(
            run_dir / f"{eid}_cycle_events.csv", index=False, encoding="utf-8-sig")


def master_row(meta, info, clip, outcome, settings, ratio, source):
    row = {"run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "app_version": __version__, "source": source}
    row.update({k: meta.get(k, "") for k in META_FIELDS})
    row.update({
        "video_file": Path(info.path).name, "video_fingerprint": info.fingerprint,
        "segment_start_sec": round(clip.start_sec, 2), "segment_end_sec": round(clip.end_sec, 2),
        "crop": clip.roi.as_text(), "variable_frame_rate": info.variable_frame_rate,
        "engine_id": outcome["engine_id"], "engine_label": outcome["label"],
        "status": "ok" if outcome.get("ok") else "failed",
        "error": outcome.get("error", ""),
        "run_seconds": round(outcome.get("seconds", 0.0), 1),
        "settings_json": json.dumps(settings, sort_keys=True, default=str),
        "assumed_movements_per_stitch": ratio,
    })
    if outcome.get("ok"):
        r = outcome["result"]
        row["method_used"] = r.method_used
        row["warnings"] = " | ".join(r.warnings)
        row["details_json"] = json.dumps(r.details, default=str)
        row.update(outcome["summary"])
        row["est_stitches"] = round(outcome["summary"]["movement_cycles"] / ratio, 2) if ratio else ""
    row["results_folder"] = str(outcome.get("run_dir", ""))
    return row


def append_master(rows):
    """Adds rows to master_results.csv. Returns a warning message or ''."""
    if not rows:
        return ""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    try:
        old = read_master()
        combined = pd.concat([old, new], ignore_index=True) if len(old) else new
        tmp = config.MASTER_CSV.with_suffix(".tmp")
        combined.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, config.MASTER_CSV)
        return ""
    except PermissionError:
        pending = config.RESULTS_DIR / f"master_results_PENDING_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        new.to_csv(pending, index=False, encoding="utf-8-sig")
        return (f"master_results.csv is open in another program (often Excel), so these rows were saved to "
                f"{pending.name}. Close Excel and use 'Merge pending files' on the Results page.")


def merge_pending():
    pend = sorted(config.RESULTS_DIR.glob("master_results_PENDING_*.csv"))
    if not pend:
        return 0
    rows = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in pend], ignore_index=True)
    msg = append_master(rows.to_dict("records"))
    if msg:
        raise PermissionError(msg)
    for p in pend:
        p.unlink()
    return len(rows)


def read_master():
    if not config.MASTER_CSV.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(config.MASTER_CSV, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


EXTERNAL_REQUIRED = ["video_file", "start_sec", "end_sec", "count"]


def import_external(df, label):
    """Adds results produced outside the app (e.g. TransRAC or ESCounts on a GPU machine)."""
    missing = [c for c in EXTERNAL_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}. Required: {', '.join(EXTERNAL_REQUIRED)}")
    rows = []
    for _, r in df.iterrows():
        dur = float(r["end_sec"]) - float(r["start_sec"])
        row = {"run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "app_version": __version__,
               "source": "external import", "video_file": r["video_file"],
               "segment_start_sec": float(r["start_sec"]), "segment_end_sec": float(r["end_sec"]),
               "engine_id": f"external:{_safe(label)}", "engine_label": f"{label} (imported)", "status": "ok",
               "method_used": str(r.get("method", f"external results: {label}")),
               "duration_sec": round(dur, 2), "movement_cycles": float(r["count"]),
               "overall_pace_per_min": round(float(r["count"]) / (dur / 60.0), 2) if dur > 0 else np.nan}
        for k in META_FIELDS + ["active_pct", "median_cycle_sec", "active_pace_per_min"]:
            if k in df.columns:
                row[k] = r[k]
        rows.append(row)
    return append_master(rows), len(rows)


def already_done(master, fingerprint, start, end, engine_id, settings):
    """True if this video segment was already analysed successfully with the same engine and settings."""
    if master is None or not len(master) or "video_fingerprint" not in master.columns:
        return False
    m = master[(master["video_fingerprint"] == fingerprint) & (master["status"] == "ok")
               & (master["engine_id"] == engine_id)
               & ((master["segment_start_sec"] - start).abs() < 0.05)
               & ((master["segment_end_sec"] - end).abs() < 0.05)]
    return bool((m["settings_json"] == json.dumps(settings, sort_keys=True, default=str)).any())


def merge_results_file(df):
    """Adds rows from another master_results.csv (e.g. produced on Colab), skipping rows
    already present. Returns (rows added, warning)."""
    if "engine_id" not in df.columns or "video_file" not in df.columns:
        raise ValueError("This does not look like a master_results.csv file.")
    key = ["run_timestamp", "video_file", "engine_id", "segment_start_sec", "segment_end_sec"]
    old = read_master()
    if len(old) and all(k in old.columns for k in key) and all(k in df.columns for k in key):
        seen = set(map(tuple, old[key].astype(str).values))
        df = df[[tuple(r) not in seen for r in df[key].astype(str).values]]
    if not len(df):
        return 0, ""
    return len(df), append_master(df.to_dict("records"))
