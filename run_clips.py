"""Runs engines on a list of clips (for example the validation clips with manual counts)
and saves the results, so they can be compared on the Validate page.

  python run_clips.py --clips manual_counts.csv --engines transrac,repnet_torch

The CSV needs video_file, start_sec and end_sec columns (the Validate page template works).
"""
import argparse
import sys

import pandas as pd

from tmbench import config
from tmbench.engines import ENGINES, engine_status
from tmbench.pipeline import parse_settings, run_engines, save_outcomes
from tmbench.video import Clip, list_videos, probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--engines", required=True)
    ap.add_argument("--folder", default=str(config.VIDEO_DIR))
    ap.add_argument("--setting", action="append", default=[], help="e.g. --setting transrac.window_sec=10")
    a = ap.parse_args()
    config.ensure_dirs()
    try:
        user_settings = parse_settings(a.setting)
    except ValueError as ex:
        sys.exit(str(ex))
    status = engine_status()
    engines = [e for e in a.engines.split(",") if e]
    for e in engines:
        if e not in ENGINES or not status[e][0]:
            sys.exit(f"Engine '{e}' is not available: {status.get(e, (False, 'unknown engine'))[1]}")
    clips = pd.read_csv(a.clips)
    vids = {v.name: v for v in list_videos(a.folder)}
    for i, r in clips.iterrows():
        v = vids.get(str(r["video_file"]))
        if v is None:
            print(f"[{i + 1}/{len(clips)}] {r['video_file']}: not found in {a.folder}")
            continue
        info = probe(v)
        start = float(r.get("start_sec", 0) or 0)
        end = float(r.get("end_sec", 0) or 0)
        end = min(end if end > start else info.duration_sec, info.duration_sec)
        meta = {k: ("" if pd.isna(r.get(k, "")) else r.get(k, "")) for k in ("artisan_id", "visit", "motif", "condition")}
        outs = run_engines(Clip(str(v), start, end), engines, user_settings, config.DEFAULT_MIN_PAUSE_SEC)
        save_outcomes(outs, meta, info, Clip(str(v), start, end), 1.0, "clips (command line)")
        print(f"[{i + 1}/{len(clips)}] {v.name} {start:.0f}-{end:.0f}s: " + "; ".join(
            f"{o['label']}: {o['summary']['movement_cycles'] if o.get('ok') else 'FAILED - ' + o['error']}" for o in outs))
    print(f"Saved to {config.MASTER_CSV}")


if __name__ == "__main__":
    main()
