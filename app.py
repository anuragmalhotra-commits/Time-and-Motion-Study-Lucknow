"""InSight Time & Motion Bench — The/Nudge Institute
High-Performance, Self-Improving AI for Chikankari Time & Motion Study.

Features:
- Ephemeral Zero-Storage video processing (stores only structured analysis, no bulky videos).
- Self-improving PyTorch Adaptive Learner that trains on verified manual counts.
- Streamlined, responsive, modern interface.
"""
import sys
from pathlib import Path

# Auto-inject virtual environment packages into sys.path
for _extra in [r"C:\repnetvenv\Lib\site-packages", r"C:\repnet_pkgs"]:
    if Path(_extra).exists() and _extra not in sys.path:
        sys.path.insert(0, _extra)

import importlib
import json
import os
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from tmbench import __version__, config, storage, validation
try:
    importlib.reload(config)
except Exception:
    pass
from tmbench.engines import ENGINES, engine_status
from tmbench.learning import (
    CHECKPOINT_PATH,
    ClipFeatures,
    extract_clip_features,
    learner,
)
from tmbench.metrics import rolling_pace
from tmbench.naming import parse_filename
from tmbench.pipeline import run_engines, save_outcomes
from tmbench.video import Clip, Roi, list_videos, preview_frame, probe, purge_old_uploads

# Storage helper functions with safe fallback
def safe_get_storage_stats():
    if hasattr(config, "get_storage_stats"):
        try:
            return config.get_storage_stats()
        except Exception:
            pass
    def _size_mb(path):
        p = Path(path)
        if not p.exists():
            return 0.0
        return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024), 2)
    return {
        "uploads_mb": _size_mb(config.UPLOAD_DIR),
        "videos_mb": _size_mb(config.VIDEO_DIR),
        "results_mb": _size_mb(config.RESULTS_DIR),
        "models_mb": _size_mb(config.MODELS_DIR),
    }

def safe_purge_all_uploads():
    if hasattr(config, "purge_all_uploads"):
        try:
            return config.purge_all_uploads()
        except Exception:
            pass
    cnt = 0
    if config.UPLOAD_DIR.exists():
        for f in config.UPLOAD_DIR.iterdir():
            try:
                if f.is_file():
                    f.unlink()
                    cnt += 1
            except OSError:
                pass
    return cnt


BROWN, BLUSH, ACCENT, ACCENT_GREEN = "#543329", "#f2eae7", "#8b4513", "#2e7d32"
ENGINE_COLOURS = ["#8b4513", "#2e7d32", "#c27c2c", "#1f77b4", "#7b1fa2", "#555555", "#d32f2f"]

st.set_page_config(
    page_title="InSight Time & Motion Bench",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Housekeeping
config.ensure_dirs()
purge_old_uploads()

# Modern, refined CSS styling
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}
.tm-header {{
    background: linear-gradient(135deg, {BROWN} 0%, #301912 100%);
    color: white;
    padding: 1.1rem 1.6rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}}
.tm-header-title {{
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
}}
.tm-header-sub {{
    font-size: 0.88rem;
    color: {BLUSH};
    margin-top: 0.25rem;
}}
.tm-badge {{
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.5rem;
}}
.tm-badge-green {{ background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }}
.tm-badge-blue {{ background: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }}
.tm-badge-amber {{ background: #fff8e1; color: #b78103; border: 1px solid #ffe082; }}
.tm-card {{
    background: #ffffff;
    border: 1px solid #e8e1dd;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 5px rgba(0,0,0,0.02);
}}
.tm-metric-card {{
    background: #faf7f5;
    border: 1px solid #e8e1dd;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}}
.tm-metric-val {{ font-size: 1.5rem; font-weight: 700; color: {BROWN}; }}
.tm-metric-lbl {{ font-size: 0.78rem; color: #777; text-transform: uppercase; letter-spacing: 0.04em; }}
</style>
<div class="tm-header">
  <div class="tm-header-title">⏱️ InSight Time &amp; Motion Bench <span style="font-weight:300; opacity:0.8; font-size:1rem;">v{__version__}</span></div>
  <div class="tm-header-sub">The/Nudge Institute · SIDBI Chikankari Pilot · AI Continuous Learning Edition</div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Caching & Helper Utilities
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_probe(path, fp_key):
    return probe(path)


@st.cache_data(show_spinner=False, max_entries=20)
def cached_preview(path, fp_key, at_sec, roi_tuple):
    return preview_frame(path, at_sec, Roi(*roi_tuple))


@st.cache_data(show_spinner=False, ttl=20)
def cached_status():
    return engine_status()


def file_key(p):
    s = Path(p).stat()
    return f"{s.st_size}-{int(s.st_mtime)}"


def get_engine_settings():
    store = st.session_state.setdefault("engine_settings", {})
    for eid, eng in ENGINES.items():
        store.setdefault(eid, eng.default_settings())
    return store


def render_engine_settings(eid, page):
    eng = ENGINES[eid]
    store = get_engine_settings()[eid]
    cols = st.columns(min(3, max(1, len(eng.settings))))
    for i, s in enumerate(eng.settings):
        col = cols[i % len(cols)]
        k = f"{page}_{eid}_{s.key}"
        cur = store.get(s.key, s.default)
        if s.kind == "float":
            store[s.key] = col.number_input(s.label, min_value=float(s.min or 0.0), max_value=float(s.max or 100.0),
                                            value=float(cur), step=float(s.step or 0.1), help=s.help or None, key=k)
        elif s.kind == "int":
            store[s.key] = int(col.number_input(s.label, min_value=int(s.min or 0), max_value=int(s.max or 100),
                                                value=int(cur), step=int(s.step or 1), help=s.help or None, key=k))
        elif s.kind == "bool":
            store[s.key] = col.checkbox(s.label, value=bool(cur), help=s.help or None, key=k)
        elif s.kind == "choice":
            opts = list(s.options)
            store[s.key] = col.selectbox(s.label, opts, index=opts.index(cur) if cur in opts else 0,
                                         help=s.help or None, key=k)
        else:
            store[s.key] = col.text_input(s.label, value=str(cur), help=s.help or None, key=k)


def engine_picker(page, default_ids=None):
    status = cached_status()
    available = [e for e in ENGINES if status.get(e, (False,))[0]]
    unavailable = [e for e in ENGINES if not status.get(e, (False,))[0]]

    # Prioritize Adaptive Learner and RepNet Torch, only keeping those available
    preferred = ["adaptive_learner", "repnet_torch", "hand_landmarks", "optical_flow"]
    defaults = [e for e in (default_ids or preferred) if e in available]
    if not defaults and available:
        defaults = [available[0]]
    elif not available:
        defaults = []

    chosen = st.multiselect(
        "Engines to run",
        available,
        default=defaults,
        format_func=lambda e: f"{ENGINES[e].label} ({ENGINES[e].kind.split()[0]})",
        key=f"{page}_engines"
    )

    if unavailable:
        with st.expander(f"⚠️ {len(unavailable)} engines currently unavailable on this machine"):
            for e in unavailable:
                st.caption(f"**{ENGINES[e].label}**: {status[e][1]}")

    if chosen:
        with st.expander("⚙️ Fine-Tune Engine Parameters (Optional)", expanded=False):
            tbs = st.tabs([ENGINES[e].label for e in chosen])
            for t, eid in zip(tbs, chosen, strict=True):
                with t:
                    st.caption(ENGINES[eid].about)
                    render_engine_settings(eid, page)
    return chosen


def meta_form(page, defaults=None):
    d = defaults or {}
    c1, c2, c3, c4 = st.columns(4)
    meta = {
        "artisan_id": c1.text_input("Artisan ID", value=d.get("artisan_id", ""), key=f"{page}_aid",
                                    placeholder="e.g. LKO-03"),
        "condition": c2.selectbox("Condition", ["With glasses", "Without glasses", "Not applicable"],
                                  index=["With glasses", "Without glasses", "Not applicable"].index(
                                      d.get("condition", "With glasses")) if d.get("condition") in [
                                      "With glasses", "Without glasses", "Not applicable"] else 0,
                                  key=f"{page}_cond"),
        "motif": c3.selectbox("Motif", ["A", "B", "Other"],
                              index=["A", "B", "Other"].index(d.get("motif", "A")) if d.get("motif") in [
                                  "A", "B", "Other"] else 0,
                              key=f"{page}_motif"),
        "visit": c4.selectbox("Visit", ["Visit 1", "Visit 2", "3-month check-in", "Bench test / practice"],
                              index=["Visit 1", "Visit 2", "3-month check-in", "Bench test / practice"].index(
                                  d.get("visit", "Visit 1")) if d.get("visit") in [
                                  "Visit 1", "Visit 2", "3-month check-in", "Bench test / practice"] else 0,
                              key=f"{page}_visit"),
    }
    with st.expander("Additional Study Metadata (Optional)", expanded=False):
        c5, c6, c7 = st.columns([1, 1, 2])
        meta["observer"] = c5.text_input("Observer initials", value=d.get("observer", ""), key=f"{page}_obs")
        om = c6.number_input("Observer stopwatch time (min)", 0.0, 600.0, float(d.get("observer_minutes") or 0.0), 0.5,
                             key=f"{page}_obsmin")
        meta["observer_minutes"] = om if om > 0 else ""
        meta["notes"] = c7.text_input("Notes (lighting, posture, interruptions)", value=d.get("notes", ""), key=f"{page}_notes")
    return meta


def outcomes_table(outcomes):
    rows = []
    for o in outcomes:
        r = {"Engine": o["label"], "Status": "✅ OK" if o.get("ok") else "❌ FAILED", "Time (s)": round(o["seconds"], 1)}
        if o.get("ok"):
            s = o["summary"]
            r.update({
                "Cycles": s["movement_cycles"],
                "Pace /min": s["active_pace_per_min"],
                "Active %": f"{s['active_pct']}%",
                "Median Cycle (s)": s["median_cycle_sec"],
                "Rhythm CV %": f"{s['cycle_cv_pct']}%",
                "Pauses": s["n_pauses"],
            })
            if "details" in o["result"].details and "predicted_cycles" in o["result"].details:
                r["AI Confidence (±)"] = f"±{o['result'].details.get('confidence_std', 0.0)}"
        else:
            r["Error"] = o.get("error", "")
        rows.append(r)
    return pd.DataFrame(rows)


def comparison_charts(outcomes):
    long_c, long_p = [], []
    for o in outcomes:
        if not o.get("ok"):
            continue
        r = o["result"]
        T = len(r.cum_counts)
        step = max(1, T // 1200)
        t = np.arange(T)[::step] / r.fps
        long_c.append(pd.DataFrame({"Time (s)": t, "Cumulative cycles": r.cum_counts[::step], "Engine": o["label"]}))
        long_p.append(pd.DataFrame({"Time (s)": t, "Pace (cycles/min)": rolling_pace(r.cum_counts, r.fps)[::step],
                                    "Engine": o["label"]}))
    if not long_c:
        return
    labels = [o["label"] for o in outcomes if o.get("ok")]
    colour = alt.Color("Engine:N", scale=alt.Scale(domain=labels, range=ENGINE_COLOURS[:len(labels)]))
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(
            alt.Chart(pd.concat(long_c)).mark_line(strokeWidth=2.2).encode(
                x=alt.X("Time (s):Q", title="Time (seconds)"),
                y=alt.Y("Cumulative cycles:Q", title="Cumulative Movement Cycles"),
                color=colour,
                tooltip=["Engine", "Time (s)", "Cumulative cycles"]
            ).properties(height=280, title="Cycle Trajectory Over Time"),
            width="stretch"
        )
    with c2:
        st.altair_chart(
            alt.Chart(pd.concat(long_p)).mark_line(strokeWidth=2.0).encode(
                x=alt.X("Time (s):Q", title="Time (seconds)"),
                y=alt.Y("Pace (cycles/min):Q", title="Rolling Pace (cycles/min)"),
                color=colour,
                tooltip=["Engine", "Time (s)", "Pace (cycles/min)"]
            ).properties(height=280, title="Rhythm & Pace Dynamics"),
            width="stretch"
        )


# ----------------------------------------------------------------------------
# Sidebar Navigation & System Badges
# ----------------------------------------------------------------------------
st.sidebar.markdown("### Navigation")
PAGES = [
    "🎬 Analyse a Clip",
    "⚡ Batch Analysis",
    "🧠 AI Learning & Validation",
    "📊 Results & Vault",
    "⚙️ Storage & Settings"
]
page = st.sidebar.radio("Go to", PAGES)

# Status badges in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st_learner_ckpt = CHECKPOINT_PATH.exists()
reg_size = len(learner.read_registry())
if st_learner_ckpt:
    st.sidebar.markdown(f'<span class="tm-badge tm-badge-green">AI Model Fine-Tuned</span><br>'
                        f'<small>Trained on <b>{reg_size}</b> validated clips</small>', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<span class="tm-badge tm-badge-amber">AI Model Baseline</span><br>'
                        f'<small>{reg_size} validated clips ready to train</small>', unsafe_allow_html=True)

storage_stats = safe_get_storage_stats()
st.sidebar.caption(f"💾 Videos cached: {storage_stats['uploads_mb'] + storage_stats['videos_mb']:.1f} MB")
st.sidebar.caption(f"📊 Analysis Vault: {storage_stats['results_mb']:.1f} MB")


# ----------------------------------------------------------------------------
# PAGE 1: Analyse a Clip (Ephemeral Zero-Storage Mode Supported)
# ----------------------------------------------------------------------------
def page_analyse():
    st.subheader("🎬 Analyse Video Clip")

    # Storage mode selector
    c_src, c_mode = st.columns([2, 1])
    with c_src:
        src = st.radio("Video Source", ["Select from videos folder", "Upload new video file"], horizontal=True)
    with c_mode:
        ephemeral = st.checkbox(
            "🔒 Zero-Storage Mode",
            value=True,
            help="Immediately purges the video file after analysis is saved. Retains 100% of structured results, CSVs, metrics, and thumbnails without taking disk space."
        )

    path, display_name = None, None
    is_uploaded = False

    if src == "Select from videos folder":
        vids = list_videos(config.VIDEO_DIR)
        if not vids:
            st.info(f"No videos found in {config.VIDEO_DIR}. Copy videos there or switch to 'Upload new video file'.")
            return
        path = st.selectbox(
            "Select Video",
            [str(v) for v in vids],
            format_func=lambda p: f"{Path(p).name} ({Path(p).stat().st_size / (1024*1024):.1f} MB)"
        )
        display_name = Path(path).name
    else:
        up = st.file_uploader(
            "Upload Video Clip",
            type=[e.strip(".") for e in config.VIDEO_EXTENSIONS],
            help="Uploaded video will be processed and automatically deleted if Zero-Storage Mode is active."
        )
        if up is None:
            st.caption("Upload an MP4, MOV or AVI file from phone or field camera.")
            return
        is_uploaded = True
        key = f"{up.name}|{up.size}"
        cache = st.session_state.get("upload_cache")
        if not cache or cache["key"] != key or not Path(cache["path"]).exists():
            if cache and Path(cache["path"]).exists():
                try:
                    Path(cache["path"]).unlink()
                except OSError:
                    pass
            dest = config.UPLOAD_DIR / f"{int(time.time())}_{storage._safe(Path(up.name).stem)}{Path(up.name).suffix.lower()}"
            with open(dest, "wb") as fh:
                up.seek(0)
                while True:
                    chunk = up.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            st.session_state["upload_cache"] = {"key": key, "path": str(dest)}
        path, display_name = st.session_state["upload_cache"]["path"], up.name

    try:
        info = cached_probe(path, file_key(path))
    except Exception as e:
        st.error(f"Could not read video: {e}")
        return

    # Video summary cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration", f"{info.duration_sec / 60:.1f} min ({info.duration_sec:.1f}s)")
    m2.metric("Frame Rate", f"{info.fps:.1f} FPS")
    m3.metric("Resolution", f"{info.width}×{info.height}")
    m4.metric("Storage Policy", "Ephemeral (Purge)" if (ephemeral and is_uploaded) else "Local Retained")

    # Segment and Crop
    st.markdown("#### 1. Segment & Working Area")
    dur = max(0.5, float(info.duration_sec))
    seg = st.slider("Segment range (seconds)", 0.0, dur, (0.0, min(dur, 60.0)), step=0.5)

    with st.expander("✂️ Crop to Working Hand (Improves Accuracy)", expanded=False):
        cc = st.columns(4)
        left = cc[0].slider("Left %", 0, 80, 10, key="roi_l")
        right = cc[1].slider("Right %", 20, 100, 90, key="roi_r")
        top = cc[2].slider("Top %", 0, 80, 10, key="roi_t")
        bottom = cc[3].slider("Bottom %", 20, 100, 90, key="roi_b")
        if right <= left + 5 or bottom <= top + 5:
            roi = Roi()
        else:
            roi = Roi(left / 100.0, top / 100.0, right / 100.0, bottom / 100.0)

    roi = roi if "roi" in locals() else Roi()
    img = cached_preview(path, file_key(path), seg[0], (roi.left, roi.top, roi.right, roi.bottom))
    if img is not None:
        st.image(img, caption=f"Preview Frame at {seg[0]:.1f}s with crop box", width=460)

    # Session metadata (with auto-extraction from filename)
    st.markdown("#### 2. Session Context")
    parsed_meta = parse_filename(Path(display_name).stem)
    meta = meta_form("an", defaults=parsed_meta)

    # Ratio & engine selection
    st.markdown("#### 3. Counting Engines")
    chosen = engine_picker("an")
    min_pause = st.number_input("Minimum pause duration (s)", 0.5, 15.0, config.DEFAULT_MIN_PAUSE_SEC, 0.5)
    ratio = st.number_input("Assumed movements per stitch", 0.5, 5.0, 1.0, 0.05,
                            help="When using AI Adaptive Learner, the stitch multiplier is also estimated automatically.")

    run_btn = st.button("🚀 Run Analysis", type="primary", disabled=not chosen)

    settings = {e: dict(get_engine_settings()[e]) for e in chosen}
    signature = json.dumps([path, seg, roi.as_text(), chosen, settings, meta, ratio, min_pause], default=str)

    if run_btn:
        clip = Clip(path, float(seg[0]), float(seg[1]), roi)
        bar = st.progress(0.0, text="Initializing engines...")
        outcomes = run_engines(clip, chosen, settings, min_pause,
                               progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
        bar.progress(1.0, text="Saving structured analysis...")

        # If zero-storage mode is active for uploaded video, delete the video file after saving results
        del_video = ephemeral and is_uploaded
        try:
            run_dir, warn = save_outcomes(outcomes, meta, info, clip, ratio, "analyse", delete_source_video=del_video)
            if del_video and Path(path).exists():
                try:
                    Path(path).unlink()
                except Exception:
                    pass
                st.session_state.pop("upload_cache", None)
        except Exception as e:
            run_dir, warn = None, f"Results could not be saved: {e}"

        bar.empty()
        st.session_state["analysis"] = {
            "signature": signature,
            "path": path,
            "display_name": display_name,
            "outcomes": outcomes,
            "run_dir": str(run_dir) if run_dir else "",
            "warn": warn,
            "features": extract_clip_features(clip, meta, outcomes),
            "meta": meta,
            "seg": seg,
            "video_purged": del_video
        }

    a = st.session_state.get("analysis")
    if not a or a.get("display_name") != display_name:
        return

    st.markdown("---")
    st.markdown("### 4. Analysis Results")
    if a.get("video_purged"):
        st.success("✅ **Zero-Storage Active:** Video file was purged from disk. All frame CSVs, pace dynamics, pauses, and metrics are preserved.")
    if a["run_dir"]:
        st.caption(f"📁 Analysis stored in: `{a['run_dir']}`")

    outcomes = a["outcomes"]
    st.dataframe(outcomes_table(outcomes), hide_index=True, width="stretch")

    comparison_charts(outcomes)

    # Detailed drilldown
    st.markdown("#### Engine Diagnostics")
    d_tabs = st.tabs([o["label"] for o in outcomes])
    for t, o in zip(d_tabs, outcomes, strict=True):
        with t:
            if not o.get("ok"):
                st.error(o.get("error", "Engine failed"))
                continue
            r = o["result"]
            st.caption(f"Method used: {r.method_used}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Detected Pauses**")
                st.dataframe(o["pauses"], hide_index=True, width="stretch")
            with c2:
                st.markdown("**Pace Progression (by fifth)**")
                st.dataframe(o["fifths"], hide_index=True, width="stretch")

            # Quick Online Learning Verification Card
            st.markdown("---")
            with st.expander("🧠 Verify Count & Teach the AI Model (Active Learning)", expanded=False):
                st.markdown("Enter the true manual count for this clip to continuously fine-tune the PyTorch model:")
                c_vc1, c_vc2 = st.columns([2, 1])
                ver_count = c_vc1.number_input(
                    "Manual reference count",
                    0.0, 500.0,
                    float(o["summary"]["movement_cycles"]),
                    0.5,
                    key=f"ver_{o['engine_id']}"
                )
                if c_vc2.button("💾 Save for Learning", key=f"btn_ver_{o['engine_id']}"):
                    n_reg = learner.add_ground_truth(
                        clip_label=f"{a['meta'].get('artisan_id', 'clip')}_{a['seg'][0]:.0f}-{a['seg'][1]:.0f}s",
                        video_file=display_name,
                        start_sec=a["seg"][0],
                        end_sec=a["seg"][1],
                        reference_count=ver_count,
                        features=a["features"],
                        meta=a["meta"]
                    )
                    st.success(f"Added to AI knowledge base! ({n_reg} verified clips now ready). Go to 'AI Learning & Validation' to train.")


# ----------------------------------------------------------------------------
# PAGE 2: Batch Analysis
# ----------------------------------------------------------------------------
def page_batch():
    st.subheader("⚡ Batch Analysis")
    st.caption("Process whole batches of videos with smart auto-metadata extraction and direct master CSV output.")

    folder = st.text_input("Videos Directory", value=str(config.VIDEO_DIR))
    vids = list_videos(folder)
    if not vids:
        st.info("No videos found in this folder.")
        return

    rows = [{"include": True, "video": str(v.relative_to(folder)), **parse_filename(v.stem)} for v in vids]
    table = st.data_editor(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        key="batch_table",
        column_config={
            "visit": st.column_config.SelectboxColumn(options=["", "Visit 1", "Visit 2", "3-month check-in", "Bench test / practice"]),
            "condition": st.column_config.SelectboxColumn(options=["", "With glasses", "Without glasses", "Not applicable"]),
            "motif": st.column_config.SelectboxColumn(options=["", "A", "B", "Other"])
        }
    )

    c_m1, c_m2 = st.columns(2)
    mode = c_m1.radio("Analysis Segment", ["Whole video", "First N minutes", "Skip initial N seconds"], horizontal=True)
    n_val = c_m2.number_input("N (seconds or minutes)", 0.0, 600.0, 5.0 if mode == "First N minutes" else 10.0, 1.0) if mode != "Whole video" else 0.0

    chosen = engine_picker("ba")
    skip_done = st.checkbox("Skip already processed videos with identical settings", value=True)
    ratio = st.number_input("Movements per stitch", 0.5, 5.0, 1.0, 0.05, key="ba_ratio")

    todo = table[table["include"]]
    if st.button(f"Start Batch ({len(todo)} videos)", type="primary", disabled=not chosen or len(todo) == 0):
        master = storage.read_master()
        bar = st.progress(0.0)
        log = []
        for i, (_, row) in enumerate(todo.iterrows()):
            vpath = str(Path(folder) / row["video"])
            try:
                info = probe(vpath)
            except Exception as e:
                log.append({"video": row["video"], "status": f"Open error: {e}"})
                continue

            if mode == "First N minutes":
                start, end = 0.0, min(info.duration_sec, n_val * 60.0)
            elif mode == "Skip initial N seconds":
                start, end = min(n_val, info.duration_sec), info.duration_sec
            else:
                start, end = 0.0, info.duration_sec

            settings = {e: dict(get_engine_settings()[e]) for e in chosen}
            engines_here = list(chosen)
            if skip_done:
                engines_here = [e for e in chosen if not storage.already_done(
                    master, info.fingerprint, start, end, e, {**ENGINES[e].default_settings(), **settings[e]})]

            if not engines_here:
                log.append({"video": row["video"], "status": "Skipped (already done)"})
                continue

            meta = {k: row.get(k, "") for k in ("artisan_id", "visit", "motif", "condition")}
            clip = Clip(vpath, start, end)
            outs = run_engines(clip, engines_here, settings, config.DEFAULT_MIN_PAUSE_SEC,
                               progress=lambda f, t, i=i, name=row["video"]: bar.progress(
                                   min((i + f) / len(todo), 1.0), text=f"[{name}] {t}"))
            run_dir, warn = save_outcomes(outs, meta, info, clip, ratio, "batch")
            log.append({
                "video": row["video"],
                "status": ", ".join(f"{o['label']}: {o['summary']['movement_cycles'] if o.get('ok') else 'FAIL'}" for o in outs)
            })
        bar.empty()
        st.success("Batch processing complete.")
        st.dataframe(pd.DataFrame(log), hide_index=True, width="stretch")


# ----------------------------------------------------------------------------
# PAGE 3: AI Learning & Validation
# ----------------------------------------------------------------------------
def page_validate():
    st.subheader("🧠 Continuous AI Learning & Model Validation")
    st.markdown("""
    This module implements **active continuous learning**:
    1. Enter manual counts for sample clips (or upload a CSV).
    2. Train / fine-tune the **PyTorch Neural Calibrator** with 1 click.
    3. The model updates its neural weights, learns artisan-specific stitch rhythms, and progressively achieves higher accuracy and ICC agreement.
    """)

    # AI Model Training Card
    reg = learner.read_registry()
    st.markdown("#### 1. PyTorch Model Health & Knowledge Base")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verified Clips in Buffer", len(reg))
    c2.metric("Checkpoint Status", "Fine-Tuned" if CHECKPOINT_PATH.exists() else "Pre-trained Baseline")

    hist = learner.get_history()
    last_mae = hist[-1]["mae"] if hist else "N/A"
    last_nmae = f"{hist[-1]['nmae_pct']}%" if hist else "N/A"
    c3.metric("Latest Error (MAE)", last_mae)
    c4.metric("Norm. Error %", last_nmae)

    # Training action
    c_tr1, c_tr2 = st.columns([2, 1])
    epochs = c_tr1.slider("Training Epochs", 50, 300, 120, 10)
    train_btn = c_tr2.button("🔥 Train / Fine-Tune PyTorch Model Now", type="primary", disabled=len(reg) < 3)

    if train_btn:
        bar = st.progress(0.0, text="Training PyTorch model on verified clips...")
        res = learner.train(epochs=epochs, progress_callback=lambda f, t: bar.progress(f, text=t))
        bar.empty()
        if res.get("ok"):
            st.success(f"🎉 Model successfully updated! New MAE: **{res['mae']}** | Error: **{res['nmae_pct']}%** across {res['n_samples']} clips.")
            st.rerun()
        else:
            st.error(res.get("error"))

    # Learning Progression Curves
    if hist:
        with st.expander("📈 View Self-Improvement Progression Curves", expanded=True):
            hist_df = pd.DataFrame(hist)
            if "mae" in hist_df.columns:
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    st.altair_chart(
                        alt.Chart(hist_df.reset_index()).mark_line(point=True, color=BROWN).encode(
                            x=alt.X("index:Q", title="Training Iteration"),
                            y=alt.Y("mae:Q", title="Mean Absolute Error (cycles)"),
                            tooltip=["timestamp", "n_samples", "mae", "nmae_pct"]
                        ).properties(height=220, title="MAE Reduction Over Iterations"),
                        width="stretch"
                    )
                with c_p2:
                    st.altair_chart(
                        alt.Chart(hist_df.reset_index()).mark_line(point=True, color=ACCENT_GREEN).encode(
                            x=alt.X("index:Q", title="Training Iteration"),
                            y=alt.Y("nmae_pct:Q", title="Normalized Error %"),
                            tooltip=["timestamp", "n_samples", "mae", "nmae_pct"]
                        ).properties(height=220, title="Normalized Error % Drop"),
                        width="stretch"
                    )

    st.markdown("---")
    st.markdown("#### 2. Enter or Upload Manual Counts for Validation")

    c_dl, c_up = st.columns(2)
    with c_dl:
        st.download_button(
            "📥 Download Blank Template (CSV)",
            validation.manual_template().to_csv(index=False),
            "manual_counts_template.csv"
        )
    with c_up:
        up = st.file_uploader("Upload Manual Counts CSV", type=["csv"], key="val_up")

    if up is not None:
        st.session_state["manual_df"] = pd.read_csv(up)

    df = st.session_state.get("manual_df", validation.manual_template())
    for c in validation.MANUAL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    disk_names = [v.name for v in list_videos(config.VIDEO_DIR)]
    existing_in_df = [str(x) for x in df["video_file"].dropna().unique() if str(x).strip() and str(x) != "nan"]
    names = sorted(list(set(disk_names + existing_in_df)))
    df = st.data_editor(
        df[validation.MANUAL_COLUMNS],
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key="manual_editor",
        column_config={
            "video_file": st.column_config.SelectboxColumn(options=names, help="Select video file"),
            "condition": st.column_config.SelectboxColumn(options=["With glasses", "Without glasses", "Not applicable"])
        }
    )
    st.session_state["manual_df"] = df

    # Helper function to extract valid reference count and clean timings
    def _extract_row_data(row, vpath):
        c1, c2 = row.get("coder1_count"), row.get("coder2_count")
        v1 = float(c1) if pd.notna(c1) and str(c1).strip() not in ("", "nan", "NaN") else np.nan
        v2 = float(c2) if pd.notna(c2) and str(c2).strip() not in ("", "nan", "NaN") else np.nan
        if not np.isnan(v1) and not np.isnan(v2):
            ref = (v1 + v2) / 2.0
        elif not np.isnan(v1):
            ref = v1
        elif not np.isnan(v2):
            ref = v2
        else:
            ref = np.nan

        info = probe(vpath)
        raw_start = row.get("start_sec")
        start_s = float(raw_start) if pd.notna(raw_start) and str(raw_start).strip() not in ("", "nan", "NaN") else 0.0
        raw_end = row.get("end_sec")
        if pd.isna(raw_end) or str(raw_end).strip() in ("", "nan", "NaN", "0", "0.0"):
            end_s = info.duration_sec
        else:
            end_s = float(raw_end)
            if end_s <= start_s or end_s > info.duration_sec:
                end_s = info.duration_sec
        return ref, start_s, end_s, info

    c_act1, c_act2 = st.columns(2)
    with c_act1:
        import_btn = st.button("📥 Import Table into AI Learning Knowledge Base")
    with c_act2:
        calib_bench_btn = st.button("⚡ 1-Click Auto-Calibrate & Benchmark", type="primary")

    if import_btn or calib_bench_btn:
        valid_rows = df.dropna(subset=["video_file"]).copy()
        added = 0
        for _, row in valid_rows.iterrows():
            v = Path(config.VIDEO_DIR) / str(row["video_file"])
            if v.exists():
                ref, start_s, end_s, info = _extract_row_data(row, str(v))
                if pd.notna(ref) and float(ref) > 0:
                    try:
                        clip = Clip(str(v), start_s, end_s)
                        feats = extract_clip_features(clip)
                        learner.add_ground_truth(
                            clip_label=str(row.get("clip_label") or v.stem),
                            video_file=v.name,
                            start_sec=start_s,
                            end_sec=end_s,
                            reference_count=float(ref),
                            features=feats,
                            meta={"condition": str(row.get("condition", "")), "artisan_id": str(row.get("artisan_id", ""))}
                        )
                        added += 1
                    except Exception as ex:
                        st.warning(f"Error processing {v.name}: {ex}")
        if added > 0:
            st.success(f"Successfully loaded {added} validated clip(s) into the knowledge base!")
            if calib_bench_btn:
                with st.spinner("Fine-tuning PyTorch Adaptive Model..."):
                    train_res = learner.train(epochs=120)
                    if train_res.get("ok"):
                        st.success(f"🎉 Model Calibrated! Learned Stitch Multiplier: **{train_res.get('learned_ratio', 1.0)}x** (Error: {train_res.get('nmae_pct', 0)}%)")
            st.rerun()

    # Run validation comparison
    st.markdown("#### 3. Run Validation Agreement Test")
    chosen = engine_picker("va", default_ids=["adaptive_learner", "repnet_torch", "hand_landmarks", "optical_flow"])
    valid = df.dropna(subset=["video_file"]).copy()
    valid = valid[valid["video_file"].astype(str).str.len() > 0]

    run_bench_btn = st.button(f"Run Agreement Benchmark ({len(valid)} clips)", disabled=not chosen or len(valid) == 0)

    if run_bench_btn or (calib_bench_btn and len(valid)):
        results = []
        bar = st.progress(0.0)
        for i, (_, row) in enumerate(valid.iterrows()):
            v = Path(config.VIDEO_DIR) / str(row["video_file"])
            if not v.exists():
                continue
            ref, start_s, end_s, info = _extract_row_data(row, str(v))
            clip = Clip(str(v), start_s, end_s)
            settings = {e: dict(get_engine_settings()[e]) for e in chosen}
            outs = run_engines(clip, chosen, settings,
                               progress=lambda f, t, i=i: bar.progress(min((i + f) / len(valid), 1.0), text=t))
            rec = {
                "clip_label": row.get("clip_label", ""),
                "video_file": row["video_file"],
                "start_sec": start_s,
                "end_sec": end_s,
                "coder1_count": row.get("coder1_count"),
                "coder2_count": row.get("coder2_count"),
                "condition": row.get("condition", ""),
            }
            for o in outs:
                rec[o["engine_id"]] = o["summary"]["movement_cycles"] if o.get("ok") else np.nan
            results.append(rec)
        bar.empty()
        out_df = pd.DataFrame(results)
        st.session_state["val_bench_df"] = out_df

    vdf = st.session_state.get("val_bench_df")
    if vdf is not None and len(vdf):
        st.markdown("##### Agreement Results Summary")
        c1v = pd.to_numeric(vdf["coder1_count"], errors="coerce")
        c2v = pd.to_numeric(vdf["coder2_count"], errors="coerce")
        vdf["reference"] = pd.concat([c1v, c2v], axis=1).mean(axis=1)

        summary_rows = []
        for e in chosen:
            if e in vdf.columns:
                agr = validation.agreement(pd.to_numeric(vdf[e], errors="coerce"), vdf["reference"])
                summary_rows.append({
                    "Engine": ENGINES[e].label,
                    "Verdict": agr.get("verdict", ""),
                    "Clips": agr.get("n_clips"),
                    "ICC(2,1) Agreement": round(agr.get("icc", np.nan), 2) if pd.notna(agr.get("icc")) else "N/A (needs ≥3 clips)",
                    "Normalized Error %": f"{agr.get('nmae_pct', np.nan)}%",
                    "Within ±10%": f"{agr.get('within_10pct', np.nan)}%",
                    "Bias": agr.get("bias", np.nan),
                })
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")
        if len(vdf) < 10:
            st.info("💡 **Statistical Note**: Official scientific validation requires $n \\ge 10$ clips to compute full 95% bootstrap confidence intervals for ICC(2,1). For small samples ($n < 10$), focus on **Normalized Error %** and **Bias** to verify engine calibration.")


# ----------------------------------------------------------------------------
# PAGE 4: Results & Vault
# ----------------------------------------------------------------------------
def page_results():
    st.subheader("📊 Analysis Vault & Reports")
    m = storage.read_master()
    if not len(m):
        st.info("No analysis records found yet. Run an analysis or batch first.")
        return

    # Filters
    c1, c2, c3 = st.columns(3)
    eng = c1.multiselect("Filter by Engine", sorted(m["engine_label"].dropna().unique()))
    art = c2.multiselect("Filter by Artisan", sorted(m["artisan_id"].dropna().astype(str).unique()))
    only_ok = c3.checkbox("Successful runs only", value=True)

    f = m.copy()
    if eng:
        f = f[f["engine_label"].isin(eng)]
    if art:
        f = f[f["artisan_id"].astype(str).isin(art)]
    if only_ok:
        f = f[f["status"] == "ok"]

    st.dataframe(f, hide_index=True, width="stretch")
    st.download_button(
        "📥 Download Filtered Results (CSV)",
        f.to_csv(index=False).encode("utf-8-sig"),
        "filtered_results.csv"
    )

    # Glasses vs No Glasses Comparative Insights
    st.markdown("---")
    st.markdown("#### 👓 Glasses vs No Glasses Impact Analysis")
    st.caption("Pairwise within-artisan comparison across verified sessions.")

    metrics = [c for c in ["active_pace_per_min", "overall_pace_per_min", "movement_cycles", "active_pct"] if c in f.columns]
    if metrics:
        metric = st.selectbox("Compare Metric", metrics)
        comp_rows = []
        for label, g in f[f["condition"].isin(["With glasses", "Without glasses"])].groupby("engine_label"):
            p = g.groupby(["artisan_id", "condition"])[metric].mean().unstack()
            if {"With glasses", "Without glasses"} <= set(p.columns):
                d = (p["With glasses"] - p["Without glasses"]).dropna()
                n = len(d)
                if n >= 2:
                    from scipy import stats
                    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
                    comp_rows.append({
                        "Engine": label,
                        "Artisans": n,
                        f"Mean Δ ({metric})": round(d.mean(), 2),
                        "95% CI": f"{d.mean() - half:.2f} to {d.mean() + half:.2f}"
                    })
                elif n == 1:
                    comp_rows.append({
                        "Engine": label,
                        "Artisans": 1,
                        f"Mean Δ ({metric})": round(d.mean(), 2),
                        "95% CI": "Single artisan"
                    })
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, width="stretch")
        else:
            st.caption("Needs paired data from the same artisans with and without glasses.")


# ----------------------------------------------------------------------------
# PAGE 5: Storage & Settings
# ----------------------------------------------------------------------------
def page_settings():
    st.subheader("⚙️ Storage, Privacy & Engine Health")

    # Storage Management Card
    st.markdown("#### 1. Zero-Storage Video Cleaner")
    st.markdown("""
    Videos can consume gigabytes of disk space. With **Zero-Storage Mode**, only the mathematical signals,
    frame metrics, pause distributions, and final CSV summaries are retained.
    """)

    stats = safe_get_storage_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Uploads Cache", f"{stats['uploads_mb']:.1f} MB")
    c2.metric("Videos Folder", f"{stats['videos_mb']:.1f} MB")
    c3.metric("Analysis Results", f"{stats['results_mb']:.1f} MB")

    if st.button("🧹 Purge All Ephemeral Uploads Cache Now"):
        n_purged = safe_purge_all_uploads()
        st.success(f"Purged {n_purged} temporary uploaded video files. All analysis records in Results Vault remain 100% intact.")
        st.rerun()

    st.markdown("---")
    st.markdown("#### 2. Engine Installation & Checkpoint Status")
    if st.button("🔄 Re-check Engine Health"):
        cached_status.clear()
        st.rerun()

    status = cached_status()
    for eid, eng in ENGINES.items():
        ok, msg = status[eid]
        with st.expander(f"{'✅' if ok else '⚪'} {eng.label} — {eng.kind}", expanded=ok):
            st.write(eng.about)
            st.caption(f"**Diagnostic Status:** {msg}")


# Routing
{
    "🎬 Analyse a Clip": page_analyse,
    "⚡ Batch Analysis": page_batch,
    "🧠 AI Learning & Validation": page_validate,
    "📊 Results & Vault": page_results,
    "⚙️ Storage & Settings": page_settings,
}[page]()
