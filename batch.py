"""Batch analysis without the browser (for long or overnight runs).

Examples (Windows, from this folder):
  Run_Batch.bat
  .venv\\Scripts\\python.exe batch.py --engines optical_flow,hand_landmarks
  .venv\\Scripts\\python.exe batch.py --folder D:\\fieldwork\\videos --engines repnet_torch --minutes 10

Session details are read from file names like LKO03_V1_G_A.mp4. Results are added to
results/master_results.csv; videos already analysed with the same settings are skipped.
"""
import argparse
import sys
import time

from tmbench import config, storage
from tmbench.engines import ENGINES, engine_status
from tmbench.naming import parse_filename
from tmbench.pipeline import parse_settings, run_engines, save_outcomes
from tmbench.video import Clip, list_videos, probe


def main():
    ap = argparse.ArgumentParser(description="InSight Time & Motion Bench - batch analysis")
    ap.add_argument("--folder", default=str(config.VIDEO_DIR))
    ap.add_argument("--engines", default="optical_flow,hand_landmarks,repnet_torch",
                    help="Comma-separated: " + ", ".join(ENGINES))
    ap.add_argument("--minutes", type=float, default=0, help="Analyse only the first N minutes (0 = whole video)")
    ap.add_argument("--skip-sec", type=float, default=0, help="Skip the first N seconds")
    ap.add_argument("--ratio", type=float, default=1.0, help="Assumed movements per stitch")
    ap.add_argument("--no-skip", action="store_true", help="Re-run videos already analysed")
    ap.add_argument("--setting", action="append", default=[],
                    help="Engine setting, e.g. --setting transrac.window_sec=10 --setting repnet_torch.strides=2")
    a = ap.parse_args()

    config.ensure_dirs()
    try:
        user_settings = parse_settings(a.setting)
    except ValueError as ex:
        sys.exit(str(ex))
    status = engine_status()
    engines = []
    for e in [x.strip() for x in a.engines.split(",") if x.strip()]:
        if e not in ENGINES:
            sys.exit(f"Unknown engine '{e}'. Choose from: {', '.join(ENGINES)}")
        if not status[e][0]:
            print(f"[skip engine] {ENGINES[e].label}: {status[e][1]}")
        else:
            engines.append(e)
    if not engines:
        sys.exit("No usable engines.")
    vids = list_videos(a.folder)
    print(f"{len(vids)} videos in {a.folder}; engines: {', '.join(ENGINES[e].label for e in engines)}")
    master = storage.read_master()
    t_all = time.time()
    for i, v in enumerate(vids, 1):
        try:
            info = probe(v)
        except Exception as ex:
            print(f"[{i}/{len(vids)}] {v.name}: could not open ({ex})")
            continue
        start = min(a.skip_sec, info.duration_sec)
        end = min(info.duration_sec, start + a.minutes * 60) if a.minutes > 0 else info.duration_sec
        todo = engines if a.no_skip else [e for e in engines if not storage.already_done(
            master, info.fingerprint, start, end, e, {**ENGINES[e].default_settings(), **user_settings.get(e, {})})]
        if not todo:
            print(f"[{i}/{len(vids)}] {v.name}: already done")
            continue
        meta = parse_filename(v.stem)
        clip = Clip(str(v), start, end)
        t0 = time.time()
        outs = run_engines(clip, todo, user_settings, config.DEFAULT_MIN_PAUSE_SEC)
        run_dir, warn = save_outcomes(outs, meta, info, clip, a.ratio, "batch (command line)")
        res = "; ".join(f"{o['label']}: {o['summary']['movement_cycles'] if o.get('ok') else 'FAILED - ' + o['error']}"
                        for o in outs)
        print(f"[{i}/{len(vids)}] {v.name} ({time.time() - t0:.0f}s): {res}")
        if warn:
            print("   " + warn)
    print(f"Finished in {(time.time() - t_all) / 60:.1f} min. Results: {config.MASTER_CSV}")


if __name__ == "__main__":
    main()
