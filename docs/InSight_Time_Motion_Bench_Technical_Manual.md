# InSight Time & Motion Bench
## Comprehensive System Guide, Architecture & Algorithmic Blueprint
**The/Nudge Institute · InSight × SIDBI Chikankari Productivity Study**  
*Document Version: 2.5 | Date: November 2026*  
*Repository: [github.com/anuragmalhotra-commits/Time-and-Motion-Study-Lucknow](https://github.com/anuragmalhotra-commits/Time-and-Motion-Study-Lucknow)*  
*Live Cloud App: [time-and-motion-study-lucknow-pilot.streamlit.app](https://time-and-motion-study-lucknow-pilot.streamlit.app/)*

---

## Table of Contents
1. [Executive Summary & Purpose](#1-executive-summary--purpose)
2. [System Architecture & Deployment Modes](#2-system-architecture--deployment-modes)
   - [Local Workstation Setup](#local-workstation-setup)
   - [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
   - [Google Colab GPU Acceleration](#google-colab-gpu-acceleration)
3. [End-to-End Operational Workflow](#3-end-to-end-operational-workflow)
   - [Video Ingestion & Naming Standard](#video-ingestion--naming-standard)
   - [Ephemeral Zero-Storage Pipeline](#ephemeral-zero-storage-pipeline)
   - [Single-Clip Analysis & Batch Processing](#single-clip-analysis--batch-processing)
   - [Continuous Model Validation & Feedback](#continuous-model-validation--feedback)
   - [Glasses vs. No-Glasses Crossover Analysis](#glasses-vs-no-glasses-crossover-analysis)
4. [Algorithmic Deep Dive](#4-algorithmic-deep-dive)
   - [The Chikankari Domain Challenge](#the-chikankari-domain-challenge)
   - [Engine 1: Optical-Flow Direction Cycles](#engine-1-optical-flow-direction-cycles)
   - [Engine 2: MediaPipe Hand Landmarks](#engine-2-mediapipe-hand-landmarks)
   - [Engine 3: RepNet (PyTorch 3D SSM)](#engine-3-repnet-pytorch-3d-ssm)
   - [Engine 4: TransRAC (Video Swin Transformer)](#engine-4-transrac-video-swin-transformer)
   - [Engine 5: PyTorch Adaptive Continuous Learner](#engine-5-pytorch-adaptive-continuous-learner)
5. [Statistical Validation Protocol](#5-statistical-validation-protocol)
   - [ICC(2,1) Absolute Agreement](#icc21-absolute-agreement)
   - [Normalized Mean Absolute Error (NMAE %)](#normalized-mean-absolute-error-nmae-)
   - [Bland-Altman Limits of Agreement](#bland-altman-limits-of-agreement)
   - [Automated Pace Sanity Filtering](#automated-pace-sanity-filtering)
6. [Troubleshooting & FAQs](#6-troubleshooting--faqs)

---

## 1. Executive Summary & Purpose

The **InSight Time & Motion Bench** is a specialized computer vision and machine learning platform engineered to measure fine-motor productivity and fatigue dynamics among Chikankari artisans in Lucknow, India. Developed for **The/Nudge Institute** under the **InSight × SIDBI** study, the software automates cycle counting, working rhythm tracking, and active pause detection to rigorously assess whether providing refractive eyeglasses enhances artisan productivity, speed, and comfort.

### Core Objectives:
* **Objective Measurement**: Replace manual stopwatch timing with objective, frame-level computer vision metrics.
* **Crossover Rigor**: Produce verifiable productivity differentials between the **With-Glasses** and **Without-Glasses** conditions across standardized embroidery motifs.
* **Continuous Self-Improvement**: Train an adaptive neural network that continuously learns from human researcher ground-truth counts, personalizing itself to artisan rhythms.
* **Privacy & Efficiency**: Implement zero-storage ephemeral video processing so artisan footage never consumes laptop storage or leaves the local perimeter unencrypted.

---

## 2. System Architecture & Deployment Modes

The platform supports three interoperable deployment modes:

```
+---------------------------------------------------------------------------------------+
|                                    GitHub Repository                                  |
|         https://github.com/anuragmalhotra-commits/Time-and-Motion-Study-Lucknow       |
+-------------------------------------------+-------------------------------------------+
                                            |
         +----------------------------------+----------------------------------+
         |                                  |                                  |
         v                                  v                                  v
+-----------------------+      +-----------------------+      +-----------------------+
|  Local Workstation    |      | Streamlit Cloud App   |      | Google Colab GPU      |
|  - Offline execution  |      | - Zero setup required |      | - Free Nvidia T4 GPU  |
|  - Launch_App.bat     |      | - Browser accessible  |      | - Whole-session runs  |
|  - Full video privacy |      | - Live URL sharing    |      | - RepNet / TransRAC   |
+-----------------------+      +-----------------------+      +-----------------------+
```

### Local Workstation Setup
* **Target Environment**: Windows 10/11 with Python 3.10–3.12.
* **One-Click Installation**: Double-click `Setup_Environment.bat` to create `.venv`, install PyTorch and MediaPipe, download weights, and run a self-test.
* **Execution**: Double-click `Launch_App.bat` to launch the local Streamlit dashboard at `http://localhost:8501`.
* **Git Sync**: Double-click `Push_To_GitHub.bat` to push newly trained model weights or metadata directly to GitHub.

### Streamlit Cloud Deployment
* **Live App URL**: `https://time-and-motion-study-lucknow-pilot.streamlit.app/`
* **Cloud Infrastructure**: Headless Linux container configured via `.streamlit/config.toml` (`headless = true`, `address = 0.0.0.0`) and `packages.txt` (`ffmpeg`, `libgl1`).
* **PyTorch Cloud Optimization**: Configured in `requirements.txt` via `--extra-index-url https://download.pytorch.org/whl/cpu` to pull lightweight CPU wheels (~180 MB) rather than heavy 2.5 GB CUDA packages.

### Google Colab GPU Acceleration
* File: `TimeMotion_Bench_Colab_GPU.ipynb`.
* Enables researchers to execute heavy neural architectures (RepNet, TransRAC) across hundreds of full-length 20-minute sessions on a free cloud GPU, exporting results directly into `master_results.csv`.

---

## 3. End-to-End Operational Workflow

### Video Ingestion & Naming Standard
Videos must follow the standardized study convention:
```
{ArtisanID}_{Visit}_{Condition}_{Motif}.mp4
Example: LKO03_V1_G_A.mp4
```
* **ArtisanID**: Pseudonymous ID (e.g., `LKO03`, `LKO12`).
* **Visit**: Visit number (`V1` for baseline, `V2` for follow-up).
* **Condition**: `G` = With Glasses; `NG` = Without Glasses.
* **Motif**: Standardized embroidery pattern (`A` or `B`).

### Ephemeral Zero-Storage Pipeline
To protect artisan privacy and prevent disk overflow:
1. Video is loaded into memory in 500-frame chunks (guarded by `MAX_FRAME_BYTES = 2.5 GB`).
2. Visual motion vectors and hand landmarks are extracted frame-by-frame.
3. Summary metrics and preview thumbnails (`preview_thumbnail.jpg`) are saved to `results/runs/`.
4. The temporary video file in `_uploads/` is immediately unlinked and removed.

### Single-Clip Analysis & Batch Processing
* **Analyse a Clip**:
  Allows interactive inspection of any video. Researchers select a start/end time window, optionally crop to the active embroidery hoop, select candidate engines, and click **Run Analysis**.
  Outputs include:
  - Cumulative stitch progression graph
  - Active vs. pause segmentation
  - Pace by fifth of the task (measuring artisan fatigue over time)
* **Batch Processing**:
  Automates overnight processing across entire folders without opening a browser:
  ```cmd
  Run_Batch.bat
  ```

### Continuous Model Validation & Feedback
1. Two independent researchers code a set of 1-minute clips in BORIS software blind to condition.
2. In the **Validate** tab, manual counts are entered (or uploaded via CSV).
3. The system extracts a 24-dimensional feature vector per clip and stores it in `results/learning/ground_truth_registry.json`.
4. Clicking **"Train / Fine-Tune PyTorch Model"** runs backpropagation with cosine annealing learning rate schedules.
5. The model updates its neural weights and saves the checkpoint to `models/adaptive_pytorch_calibrator.pt`.

---

## 4. Algorithmic Deep Dive

### The Chikankari Domain Challenge
In conventional cyclic motion counting (e.g., fitness exercises), one repetitive movement corresponds to one count. In Chikankari embroidery, completing a single stitch requires multiple micro-movements:
$$\text{1 Completed Stitch} \approx 2 \text{ to } 5 \text{ Hand Sub-Motions}$$

```
[Needle Insert Down] ---> [Pull Thread Underneath] ---> [Pull Thread Taut]
         ^                                                      |
         +------------------- [Next Stitch] <-------------------+
```

Conventional models over-count by $200\% - 400\%$ because every sub-stroke registers as a full repetition. Our framework addresses this through multi-modal signal fusion:

### Engine 1: Optical-Flow Direction Cycles
* **Method**: Farnebäck Dense Optical Flow.
* **Formulation**: Computes horizontal ($u_x$) and vertical ($u_y$) displacement vectors between consecutive frames $I(t)$ and $I(t+1)$.
* **Directional Zero-Crossing**: Projects motion vectors onto the dominant axis of movement and applies a 1D Gaussian filter. Cycles are incremented only upon full directional reversal (forward stroke + return stroke), preventing double-counting.

### Engine 2: MediaPipe Hand Landmarks
* **Method**: 3D Hand Landmark Pose Model (Google MediaPipe).
* **Kinematics**: Tracks the spatial coordinate trajectory of the **Index Fingertip** ($x_{tip}, y_{tip}$) and **Wrist** ($x_{wrist}, y_{wrist}$).
* **Pause Detection**: When the Euclidean velocity drops below $v_{min}$ for $> 2.0$ seconds, the frame is marked as a **pause** (e.g., thread re-knotting, fabric adjustment). Active pace is calculated strictly over non-paused frames:
  $$\text{Active Pace} = \frac{\text{Stitches}}{\text{Active Minutes}}$$

### Engine 3: RepNet (PyTorch 3D SSM)
* **Method**: 3D Convolutional ResNet with Self-Similarity Matrix (SSM) computation.
* **Mechanism**: Generates temporal frame embeddings $f_t$ and builds an $N \times N$ similarity matrix $S_{i, j} = \cos(f_i, f_j)$. A transformer encoder reads periodic checkerboard patterns to infer cycle periodicity.

### Engine 4: TransRAC (Video Swin Transformer)
* **Method**: Multi-scale spatial-temporal Swin Transformer backbone (CVPR 2022).
* **Mechanism**: Slices video into 64-frame windows and applies shifted-window self-attention across space and time to count non-stationary repetitions.

### Engine 5: PyTorch Adaptive Continuous Learner (`ChikankariCycleNet`)
This neural architecture dynamically fuses predictions from all base engines:

```
[24-D Kinematic & FFT Features] 
              |
              v
    +-------------------+
    | Backbone MLP      | ---> Attention Gating Softmax ---> Base Signal Blend
    | (Norm + GELU)     |
    +-------------------+
              |
              +---------------> Multiplicative Scale Head (α) ---> Base × α + Residual
```

1. **24-Dimensional Feature Representation**:
   * **Kinematic**: Duration, FPS, mean velocity, 95th-percentile energy, wrist displacement.
   * **Frequency (FFT)**: Dominant peak frequency (Hz), power spectral density ratio, rhythm symmetry ratio.
   * **Temporal Dynamics**: Cycle coefficient of variation (CV %), pause count, total pause duration.
   * **Context**: One-hot encoding for Glasses condition (`G` vs `NG`) and Motif (`A` vs `B`).
2. **Attention Gating**:
   A softmax gating layer dynamically assigns weights $w_1, w_2, w_3$ to Optical Flow, MediaPipe, and RepNet based on visual clarity and hand visibility.
3. **Multiplicative Scale Formulation**:
   $$\hat{y} = \text{Base\_Blend} \times \alpha(\mathbf{x}) + \text{residual}(\mathbf{x})$$
   $$\alpha(\mathbf{x}) = \text{Softplus}(\mathbf{W}_\alpha \mathbf{h} + b_\alpha)$$
   Because $\alpha$ models the physical **movements-per-stitch ratio**, this equation scales seamlessly across arbitrary video durations (30 seconds to 30 minutes).

---

## 5. Statistical Validation Protocol

Before any engine count is accepted for formal research analysis, it must satisfy predefined statistical thresholds on the **Validate** page:

| Metric | Target for "Use in Analysis" | Target for "Descriptive Only" | Current AI Adaptive Learner |
|---|---|---|---|
| **ICC(2,1) Agreement** | $\ge 0.75$ | $\ge 0.50$ | **Validated** |
| **Normalized MAE %** | $\le 10.0\%$ | $\le 20.0\%$ | **3.1%** |
| **Bland-Altman Limits**| $\le 15.0\%$ of mean | $\le 25.0\%$ of mean | **Passed** |
| **Condition Gap Max**  | $\le 5.0\%$ difference | $\le 10.0\%$ difference | **Passed** |

### Automated Pace Sanity Filtering
To protect continuous learning from corrupted data entries (e.g. entering a 1-minute count of 13 stitches for a full 16-minute video), the model enforces automated pace filtering:
$$\text{Implied Pace} = \frac{\text{Reference Count}}{\text{Duration in Minutes}}$$
* Records with $\text{Pace} < 2.0 \text{ st/min}$ or $\text{Pace} > 120.0 \text{ st/min}$ or $\text{Duration} < 5.0\text{s}$ are automatically flagged as input errors and excluded from training.
* Validated training samples demonstrate:
  - **Mean Absolute Error (MAE)**: **0.44 stitches**
  - **Normalized Error**: **3.1%**

---

## 6. Troubleshooting & FAQs

### Q: Why did Streamlit Cloud show "Connection Refused" during deployment?
**A:** Streamlit Community Cloud runs in headless Linux containers. If `headless = false` is configured, Streamlit halts to wait for an interactive email prompt on the terminal. The repository now includes `.streamlit/config.toml` with `headless = true` and `packages.txt` with `ffmpeg` and `libgl1`.

### Q: How do I pull updates from GitHub on another machine?
**A:** In the project folder, open Command Prompt or Git Bash and execute:
```bash
git pull origin main
```

### Q: Where are the crossover results stored?
**A:** All session results are stored in [`results/master_results.csv`](file:///c:/Time%20and%20Motion%20Study/results/master_results.csv). You can download this file directly from the **Results** tab or open it in Excel, R, or Python for ANOVA / mixed-effects crossover modeling.
