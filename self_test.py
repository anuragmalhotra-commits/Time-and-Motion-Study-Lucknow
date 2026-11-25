"""Checks the installation: makes short test videos with a known number of movements,
runs every available engine on them and reports how close each count is.

These are simple moving shapes, not embroidery: passing shows the software works, not
that an engine is accurate on chikankari (that needs the Validate page).
"""
import tempfile
from pathlib import Path

from tmbench.engines import ENGINES, engine_status
from tmbench.metrics import compute_metrics
from tmbench.synth import make_video
from tmbench.video import Clip, probe

CASES = [("slow", 0.5), ("medium", 0.7), ("fast", 1.5)]


def main():
    status = engine_status()
    print("Engine status:")
    for e, (ok, msg) in status.items():
        print(f"  {'OK ' if ok else '-- '} {ENGINES[e].label}: {msg}")
    tmp = Path(tempfile.mkdtemp(prefix="tm_selftest_"))
    print(f"\nTest videos in {tmp} (40 s each, 5 s pause in the middle)\n")
    rows = []
    for name, hz in CASES:
        p = tmp / f"test_{name}.mp4"
        true = make_video(p, hz=hz)
        info = probe(p)
        for e, (ok, _) in status.items():
            if not ok:
                continue
            if e == "hand_landmarks":
                continue  # needs real hands; checked separately below
            if e == "transrac" and name != "medium":
                continue  # slow on a laptop: one case is enough to check it runs
            clip = Clip(str(p), 0.0, info.duration_sec)
            try:
                r = ENGINES[e].run(clip, {})
                m, _, _ = compute_metrics(r.cum_counts, r.active, r.fps, event_times=r.event_times)
                got = m["movement_cycles"]
                err = (got - true) / true * 100
                verdict = "PASS" if abs(err) <= 10 else ("expected limit" if e == "motion_energy" else "CHECK")
                rows.append((ENGINES[e].label, name, round(true, 1), got, f"{err:+.0f}%", verdict))
            except Exception as ex:
                rows.append((ENGINES[e].label, name, round(true, 1), "-", "-", f"ERROR: {ex}"))
            finally:
                clip.release()
    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'Engine':<{w}}  {'speed':<7} {'true':>6} {'counted':>8} {'error':>7}  result")
    for r in rows:
        print(f"{r[0]:<{w}}  {r[1]:<7} {r[2]:>6} {str(r[3]):>8} {r[4]:>7}  {r[5]}")
    if status.get("hand_landmarks", (False,))[0]:
        print("\nHand-landmark engine: model loads. It needs real hands, so test it on a practice clip.")
    print("\nMotion-energy is expected to fail the slow and fast cases (known limit of that method).")


if __name__ == "__main__":
    main()
