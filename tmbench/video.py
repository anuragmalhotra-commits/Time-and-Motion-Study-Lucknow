"""Reading video safely and consistently.

Frames are sampled by their real timestamps (not by counting frames), so phone videos
with a variable frame rate still give correct timings. Every engine receives frames
from the same segment and the same crop, at the size and rate it needs.
"""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int
    variable_frame_rate: bool
    fps_note: str
    fingerprint: str


def fingerprint(path):
    """Short ID from file name, size and modified time (detects a changed or replaced file)."""
    p = Path(path)
    st = p.stat()
    return hashlib.sha1(f"{p.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]


def probe(path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open the video {Path(path).name}. It may be corrupted, still downloading "
                      "(for example from OneDrive), or in an unsupported format.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Check frame timing on the first ~300 frames
    stamps = []
    for _ in range(300):
        ok = cap.grab()
        if not ok:
            break
        stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    vfr, note = False, ""
    if len(stamps) > 20:
        d = np.diff(stamps)
        d = d[d > 0]
        if len(d) > 10:
            med = float(np.median(d))
            measured_fps = 1.0 / med if med > 0 else fps
            spread = float(np.percentile(d, 95) / max(np.percentile(d, 5), 1e-6))
            vfr = spread > 1.5
            if fps <= 0:
                fps = measured_fps
            if vfr:
                note = (f"Frame timing varies (variable frame rate, typical of phones). "
                        f"Timings use real timestamps. Nominal {fps:.1f} FPS.")
            elif abs(measured_fps - fps) > 1:
                note = f"Header says {fps:.1f} FPS but frames arrive at about {measured_fps:.1f} FPS; using timestamps."
    if fps <= 0:
        fps = 30.0
        note = (note + " " if note else "") + "Frame rate unknown; assumed 30 FPS."

    # Duration from the last frame's timestamp where possible
    duration = n / fps if n > 0 else 0.0
    if n > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
        if cap.grab():
            t_last = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t_last > 0:
                duration = t_last + 1.0 / fps
    cap.release()
    return VideoInfo(str(path), float(fps), n, float(duration), w, h, vfr, note, fingerprint(path))


@dataclass
class Roi:
    """Crop rectangle as fractions of the frame (0-1)."""
    left: float = 0.0
    top: float = 0.0
    right: float = 1.0
    bottom: float = 1.0

    def is_full(self):
        return (self.left, self.top, self.right, self.bottom) == (0.0, 0.0, 1.0, 1.0)

    def as_text(self):
        return "full frame" if self.is_full() else (
            f"L{self.left:.2f} T{self.top:.2f} R{self.right:.2f} B{self.bottom:.2f}")

    def apply(self, frame):
        h, w = frame.shape[:2]
        x0, x1 = int(self.left * w), max(int(self.left * w) + 2, int(self.right * w))
        y0, y1 = int(self.top * h), max(int(self.top * h) + 2, int(self.bottom * h))
        return frame[y0:y1, x0:x1]


def _fit_square(img, size):
    """Resize keeping proportions and pad to a square (nothing is cut off or stretched)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    out[:] = np.median(resized.reshape(-1, 3), axis=0).astype(np.uint8)  # neutral padding
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def read_segment(path, start_sec, end_sec, target_fps, size, roi=None, fit="pad"):
    """Returns (frames RGB uint8 [N,size,size,3], fps actually used, timestamps in seconds).

    fit = "pad": keep proportions and pad to a square (default);
          "squash": stretch to a square (what RepNet's own scripts do).
    """
    roi = roi or Roi()
    info_fps = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(min(target_fps, info_fps)) if target_fps else float(info_fps)
    n_expected = int(max(0.0, end_sec - start_sec) * fps) + 1
    if n_expected * size * size * 3 > config.MAX_FRAME_BYTES:
        raise MemoryError(
            f"This segment needs about {n_expected * size * size * 3 / 1e9:.1f} GB of memory at {size}px and "
            f"{fps:.0f} FPS (limit {config.MAX_FRAME_BYTES / 1e9:.1f} GB). Please choose a shorter segment.")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open {Path(path).name}.")
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    frames, stamps = [], []
    next_t = start_sec
    step = 1.0 / fps
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > end_sec:
            break
        if t + 1e-6 < next_t:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue
        frame = roi.apply(frame)
        if fit == "squash":
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        else:
            frame = _fit_square(frame, size)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        stamps.append(t)
        # schedule the next sample on the regular grid (skip ahead if frames were dropped)
        next_t += step
        while next_t <= t:
            next_t += step
    cap.release()
    if not frames:
        raise ValueError("No frames could be read from this segment.")
    return np.asarray(frames), fps, np.asarray(stamps)


def preview_frame(path, at_sec, roi=None, max_width=720):
    """One frame (RGB) with the crop box drawn on it, for checking the set-up."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, at_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if roi is not None and not roi.is_full():
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (int(roi.left * w), int(roi.top * h)), (int(roi.right * w), int(roi.bottom * h)),
                      (48, 61, 105), max(2, w // 300))
    scale = min(1.0, max_width / frame.shape[1])
    if scale < 1:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@dataclass
class Clip:
    """A video segment that engines analyse. Frames are read on request and cached."""
    path: str
    start_sec: float
    end_sec: float
    roi: Roi = field(default_factory=Roi)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def duration(self):
        return self.end_sec - self.start_sec

    def frames(self, target_fps, size, fit="pad"):
        key = (target_fps, size, fit)
        if key not in self._cache:
            self._cache.clear()  # keep only one resolution in memory at a time
            self._cache[key] = read_segment(self.path, self.start_sec, self.end_sec, target_fps, size, self.roi, fit)
        return self._cache[key]

    def release(self):
        self._cache.clear()


def purge_old_uploads(max_age_hours=24):
    if not config.UPLOAD_DIR.exists():
        return
    import time
    cutoff = time.time() - max_age_hours * 3600
    for f in config.UPLOAD_DIR.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def list_videos(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    out = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in config.VIDEO_EXTENSIONS and not p.name.startswith("~"):
            out.append(p)
    return out
