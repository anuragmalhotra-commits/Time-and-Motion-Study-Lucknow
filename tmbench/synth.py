"""Synthetic test videos with a known number of movement cycles."""
import cv2
import numpy as np


def make_video(path, hz=0.7, fps=30, dur=40, size=320, pause=(15, 20), asym=False, two_objects=False, seed=0):
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur((rng.random((size, size)) * 60 + 170).astype(np.uint8), (9, 9), 0)
    tex = cv2.GaussianBlur((rng.random((60, 60)) * 255).astype(np.uint8), (5, 5), 0)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    phase, prev_t = 0.0, 0.0
    for k in range(int(dur * fps)):
        t = k / fps
        moving = not (pause and pause[0] <= t < pause[1])
        if moving:
            phase += hz * (t - prev_t)
        prev_t = t
        ph = phase % 1.0
        if asym:   # quick pull (20% of cycle), slow return
            x = np.sin(np.pi * ph / 0.4) if ph < 0.2 else np.cos(np.pi * (ph - 0.2) / 1.6)
        else:
            x = np.sin(2 * np.pi * ph)
        img = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        cx = int(size / 2 + size * 0.25 * x) - 30
        img[size // 2 - 30:size // 2 + 30, cx:cx + 60] = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
        if two_objects:   # a still second object
            img[20:80, 20:80] = 60
        vw.write(img)
    vw.release()
    return phase  # true number of completed cycles
