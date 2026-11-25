# InSight Time & Motion Bench

**The/Nudge Institute · InSight × SIDBI chikankari pilot · version 2.1**

A local app for testing several video-counting engines side by side on chikankari
footage, checking them against manual counts, and producing one results file for the
glasses / no-glasses comparison. Nothing leaves the laptop: the app only accepts
connections from the computer it runs on.

> Until an engine passes validation against manual counts, every number it gives is an
> **indicator of hand movement**, not a verified stitch count.

---

## 1. Set-up (once per laptop)

1. Install **Python 3.12** from python.org and tick *Add python.exe to PATH*.
   (MediaPipe does not yet support Python 3.13.)
2. Copy this folder to a local folder, ideally **not inside OneDrive**, e.g. `C:\InSight_TimeMotion`.
3. Double-click **`Setup_Environment.bat`**. It needs internet and takes 10–20 minutes. It:
   - creates a private Python environment (`.venv`) and installs the packages;
   - downloads the MediaPipe hand model (or copies it from `C:\repnet_pkgs`);
   - downloads the RepNet PyTorch code and weights, and the TransRAC code and Video Swin weights,
     from their authors' pages (`setup_models.py`);
   - runs a self-test.
4. **TransRAC checkpoint (manual):** its authors share it only on OneDrive/Baidu. Open
   github.com/SvipRepetitionCounting/TransRAC, follow "The model ckpt has been available" (OneDrive,
   extraction code `transrac`) and put the downloaded file in `models\transrac\`.

If a download is blocked, the script prints the address so it can be downloaded by hand.

## 2. Everyday use

- Put videos in the **`videos`** folder (subfolders are fine).
  Name them `ArtisanID_Visit_Condition_Motif`, e.g. `LKO03_V1_G_A.mp4`
  (`V1`/`V2`/`M3`; `G` = with glasses, `NG` = without; motif `A`/`B`). The batch page reads these.
- Double-click **`Launch_App.bat`**; the app opens in the browser.
- **Run_Batch.bat** analyses every video without the browser (good for overnight runs).
- **Run_Self_Test.bat** checks the installation at any time.

## 3. Pages

| Page | What it does |
|---|---|
| Analyse a clip | Choose a video, a segment and an optional crop, enter session details, run one or more engines, compare them on one chart. Results are saved automatically. |
| Batch | Run engines on many videos; session details are read from file names and can be corrected first. Already-analysed videos are skipped. |
| Validate against manual counts | Enter manual counts for sample clips (two coders); the app runs the engines on exactly those clips and reports ICC, error, Bland–Altman limits and whether error differs between conditions, with a verdict against the proposed thresholds. |
| Results | All results, filters, download, and an exploratory within-artisan glasses comparison. |
| Engines & set-up | Engine status and descriptions; import results from other models (e.g. ESCounts). |

## 4. Engines

| Engine | Type | Notes |
|---|---|---|
| **RepNet (PyTorch)** | Trained model (Google, CVPR 2020) | Main candidate. Trained on general videos, not embroidery. Tries several strides; fix one stride for the study. Slowest on a laptop CPU (about one minute per minute of video per stride setting). |
| **TransRAC (official model)** | Trained model (CVPR 2022) | The published model, run without its outdated mmcv/mmaction libraries (its own code, unchanged otherwise). Trained on exercise videos. Each window (20 s by default) becomes 64 frames; shorten it for fast cycles. Slow on CPU (about 2 s per second of video) – use the Colab notebook for whole sessions. Needs the manual checkpoint download. |
| **Hand-landmark cycles (MediaPipe)** | Hand detector + rules | Follows one fingertip; counts only where the hand is seen; reports how much of the clip could be analysed. |
| **Optical-flow direction cycles** | Heuristic (new) | Uses the *direction* of movement, so it does not half- or double-count like the old method. Crop to the working hand. Fast. |
| **Motion-energy autocorrelation** | Heuristic (previous app) | Kept for comparison only. Correct only for cycles of about 0.8–1.6 s. |
| **RepNet (TensorFlow, previous set-up)** | Trained model | Runs your existing `batch_repnet.py` + checkpoint, to confirm the PyTorch version matches. Needs TensorFlow; untested by the developer. |
| **External results** | Any | Counts from other models run elsewhere (see §6). |

Self-test results on synthetic moving shapes (40 s, 5 s pause), as a software check only:
optical flow within 3% at 0.5, 0.7 and 1.5 cycles/s; motion energy +69%, −15% and −46%.
The hand engine was checked with real hand photographs moved at 0.5, 0.7 and 1.5 cycles/s
(within 3%). RepNet and TransRAC could only be checked for correct running: their trained weights could not be
downloaded in the development environment (TransRAC's Video Swin backbone loaded fully: 183 of 183 weights).
Run the self-test after set-up to see their results.

## 5. Recommended workflow for the pilot

1. **Bench test** (before fieldwork): record practice clips, run all engines, adjust the crop and recording set-up.
2. **Manual counts**: code a sample of one-minute clips in BORIS with two coders, blind to condition.
3. **Validate**: enter the counts on the Validate page; keep only engines marked *meets thresholds*
   (or *descriptive only* for supporting indicators). Fix their settings (e.g. RepNet stride) and do not change them afterwards.
4. **Analyse** all sessions with the chosen engines (Batch page or Run_Batch.bat).
5. Use `results/master_results.csv` for the crossover analysis, alongside observer-timed completion and quality scores.

## 6. GPU runs, other computers and external models

**Colab GPU.** `TimeMotion_Bench_Colab_GPU.ipynb` runs the same code on a free Colab GPU (useful for TransRAC
and RepNet on whole sessions). Bring its `master_results.csv` back with *Results → Add results from another
computer or Colab*; on the Validate page, choose them under *Also compare saved results*.

**Command line.** `batch.py` (whole videos) and `run_clips.py` (a list of clips, e.g. the validation template)
accept engine settings, for example:

```
.venv\Scripts\python.exe run_clips.py --clips manual_counts.csv --engines transrac,repnet_torch --setting transrac.window_sec=10 --setting repnet_torch.strides=2
```

**Other models (e.g. ESCounts).** Run them elsewhere and import a CSV on the *Engines & set-up* page:

```
video_file,start_sec,end_sec,count[,artisan_id,visit,motif,condition,method]
LKO03_V1_G_A.mp4,60,120,41.5,LKO03,Visit 1,A,With glasses,ESCounts zero-shot
```

## 7. Outputs

- `results/master_results.csv` — one row per engine per analysis: session details, video
  fingerprint, segment, crop, engine, exact settings, method actually used, warnings, and the indicators
  (cycles, overall and active pace, active %, median cycle time, cycle-time variability, pauses, pace change across the task).
- `results/runs/<date_time_artisan_...>/` — per-engine frame-level data, pauses, pace by fifth, cycle events.
- `results/validation/` — each validation run.

If `master_results.csv` is open in Excel, new rows go to a `PENDING` file; use *Merge pending files* on the Results page.

## 8. Privacy

- The app listens only on `localhost`; do not share it with a public link while artisan videos are on the laptop.
- Uploaded copies are deleted after 24 hours; prefer copying videos into the `videos` folder.
- Use pseudonymous artisan IDs.

## 9. Changing folders

Set environment variables before launching: `INSIGHT_VIDEO_DIR`, `INSIGHT_RESULTS_DIR`,
`INSIGHT_MODELS_DIR`, `INSIGHT_MAX_FRAME_GB` (memory limit, default 2.5).

## 10. Third-party components

Downloaded during set-up, not bundled: TransRAC code (Apache 2.0) and Video Swin weights
(Apache 2.0); the TransRAC checkpoint (from its authors; check its terms); MediaPipe and its hand model (Google, Apache 2.0);
RepNet weights converted from Google's RepNet checkpoint (Apache 2.0) by the
`materight/RepNet-pytorch` port — that repository has no licence file, so it is used here for
internal research only; check with its author before redistributing. Also used: Streamlit, OpenCV,
PyTorch, timm, einops, SciPy, pandas, Altair (all permissive licences).

## 11. Adding an engine

Create a class in `tmbench/engines/` that subclasses `Engine` (see `base.py`), returns an
`EngineResult` (cumulative count and active flag per frame), and add it to `ENGINES` in
`tmbench/engines/__init__.py`. All metrics, saving, comparison and validation then work automatically.

## 12. Working in VS Code

Open the project folder in VS Code (*File → Open Folder*) and accept the recommended extensions
(Python, Python Debugger, Pylance, Jupyter, Rainbow CSV, Ruff). After `Setup_Environment.bat` has created
`.venv`, VS Code uses it automatically (check the interpreter shown in the bottom bar).

- **Run and Debug** (Ctrl+Shift+D): *App (Streamlit)* starts the app with breakpoints working;
  also *Batch analysis*, *Run validation clips*, *Self-test*, *Download model files*, *Current Python file*.
  Engines are picked from a list when you start a run.
- **Tasks** (Ctrl+Shift+P → *Run Task*): set-up, launch app (also Ctrl+Shift+B), self-test, batch,
  validation clips, and a quick error check of all code.
- **`.env`**: optional folder settings for VS Code runs (remove the `#` to use a line).
- **`.gitignore`** keeps videos, results, models and `.env` out of any Git repository, so artisan
  footage is never committed by mistake.
- **`ruff.toml`**: code-style checks (the code currently passes them all).

In the debug configuration the app reloads when you save a file; the normal launcher does not.
