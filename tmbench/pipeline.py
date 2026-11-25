"""Runs selected engines on one clip, computes common metrics and saves everything."""
import time
import traceback

from . import storage
from .engines import ENGINES
from .metrics import compute_metrics


def run_engines(clip, engine_ids, settings_by_engine=None, min_pause_sec=2.0, progress=None):
    """Returns a list of outcome dicts (one per engine). Never raises for engine errors."""
    settings_by_engine = settings_by_engine or {}
    outcomes = []
    for k, eid in enumerate(engine_ids):
        eng = ENGINES[eid]
        s = {**eng.default_settings(), **settings_by_engine.get(eid, {})}

        def sub(frac, text, k=k, label=eng.label):
            if progress:
                progress((k + frac) / len(engine_ids), f"[{label}] {text}")

        t0 = time.time()
        out = {"engine_id": eid, "label": eng.label, "settings": s}
        try:
            r = eng.run(clip, s, progress=sub)
            summary, pauses, fifths = compute_metrics(r.cum_counts, r.active, r.fps, min_pause_sec, r.event_times)
            out.update(ok=True, result=r, summary=summary, pauses=pauses, fifths=fifths)
        except MemoryError as e:
            out.update(ok=False, error=str(e))
        except Exception as e:
            out.update(ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
        out["seconds"] = time.time() - t0
        outcomes.append(out)
    clip.release()
    return outcomes


def save_outcomes(outcomes, meta, info, clip, ratio, source, delete_source_video=False):
    run_dir = storage.make_run_dir(meta, info.path)
    # Save lightweight thumbnail of the first frame with ROI preview (persists visual context without storing video)
    try:
        from .video import preview_frame
        thumb = preview_frame(info.path, clip.start_sec, clip.roi, max_width=480)
        if thumb is not None:
            import cv2
            cv2.imwrite(str(run_dir / "preview_thumbnail.jpg"), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
    except Exception:
        pass

    rows = []
    for o in outcomes:
        o["run_dir"] = str(run_dir)
        storage.save_engine_output(run_dir, o)
        rows.append(storage.master_row(meta, info, clip, o, o["settings"], ratio, source))
    warn = storage.append_master(rows)

    if delete_source_video:
        try:
            from pathlib import Path
            p = Path(info.path)
            if p.exists():
                p.unlink()
        except Exception:
            pass

    return run_dir, warn



def parse_settings(pairs):
    """'engine.key=value' strings -> {engine: {key: typed value}} (for the command-line tools)."""
    from .engines import ENGINES
    out = {}
    for p in pairs or []:
        if "=" not in p or "." not in p.split("=", 1)[0]:
            raise ValueError(f"Setting '{p}' must look like engine.key=value, e.g. transrac.window_sec=10")
        lhs, val = p.split("=", 1)
        eid, key = lhs.split(".", 1)
        if eid not in ENGINES:
            raise ValueError(f"Unknown engine '{eid}' in '{p}'")
        spec = {s.key: s for s in ENGINES[eid].settings}.get(key)
        if spec is None:
            raise ValueError(f"Unknown setting '{key}' for {eid}. Options: {', '.join(s.key for s in ENGINES[eid].settings)}")
        if spec.kind == "float":
            v = float(val)
        elif spec.kind == "int":
            v = int(val)
        elif spec.kind == "bool":
            v = val.lower() in ("1", "true", "yes")
        elif spec.kind == "choice":
            v = type(spec.default)(val)
        else:
            v = val
        out.setdefault(eid, {})[key] = v
    return out
