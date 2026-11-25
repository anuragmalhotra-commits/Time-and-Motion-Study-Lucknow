"""Common interface for every counting engine.

To add a new engine: subclass Engine, fill in the class attributes, implement check()
and run(), and add it to ENGINES in engines/__init__.py.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Setting:
    key: str
    label: str
    kind: str              # "float", "int", "bool", "choice", "text"
    default: object
    help: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[list] = None


@dataclass
class EngineResult:
    engine_id: str
    fps: float
    cum_counts: np.ndarray            # cumulative cycles per frame
    active: np.ndarray                # bool per frame: repetitive work detected
    score: Optional[np.ndarray] = None  # 0-1 confidence/periodicity per frame, if any
    event_times: Optional[list] = None  # cycle times (s from clip start), if the engine finds them
    method_used: str = ""             # what actually ran (shows fallbacks)
    details: dict = field(default_factory=dict)    # engine-specific diagnostics
    warnings: list = field(default_factory=list)
    extra_series: dict = field(default_factory=dict)  # extra per-frame columns for export


class Engine:
    id = "base"
    label = "Base engine"
    kind = "heuristic"         # "trained model" or "heuristic"
    short = ""                 # one-line description
    about = ""                 # longer, honest description including limits
    frame_size = 112
    target_fps = 15
    fit = "pad"
    settings = []

    def check(self):
        """Returns (available: bool, message: str)."""
        return True, "Ready"

    def default_settings(self):
        return {s.key: s.default for s in self.settings}

    def run(self, clip, settings, progress=None) -> EngineResult:
        raise NotImplementedError


def report(progress, frac, text):
    if progress is not None:
        try:
            progress(float(min(max(frac, 0.0), 1.0)), text)
        except Exception:
            pass
