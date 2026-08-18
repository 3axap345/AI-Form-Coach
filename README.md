# AI Form Coach

AI Form Coach is a Python/OpenCV + MediaPipe prototype for collecting squat repetitions, preprocessing pose landmarks into a canonical skeleton format, running a binary form classifier, and displaying feedback in a live camera HUD.

The project currently focuses on:

- live MediaPipe pose collection;
- automatic squat repetition detection;
- canonical skeleton samples shaped `[60, 12, 4]`;
- binary `correct` / `incorrect` squat classification;
- lightweight rule-based form diagnostics;
- UI-PRMD conversion utilities for offline training.

> This project provides coaching heuristics and ML prototype feedback. It is not a medical diagnostic tool.

## AI-Assisted Development Workflow

This project was developed with an AI-assisted workflow. Architecture, product direction, technical decisions, and integration choices were guided and reviewed by the project owner, while AI coding assistants were used to accelerate routine implementation, debugging, research, and documentation tasks.

Claude was used during the initial collector phase to speed up implementation of the live squat data capture prototype:

- scaffolded the core collector modules `config.py`, `camera.py`, `pose.py`, `landmarks.py`, `repetition.py`, `preprocessing.py`, `quality.py`, `storage.py`, `ui.py`, and `main.py`;
- assisted with webcam capture, MediaPipe Pose processing, selected squat landmarks, knee-angle repetition detection, 60-frame resampling, pelvis-centered normalization, quality checks, async `.npy` + `.json` saving, and undo support;
- helped investigate local setup issues around Python 3.14 incompatibility with MediaPipe, moving to Python 3.11 + venv, and path issues affecting MediaPipe model assets;
- supported research into rehabilitation / exercise datasets, UI-PRMD availability, and dataset alternatives.

Codex was used in the follow-up integration phase to help turn the prototype into a more cohesive training/inference project:

- audited the collector, UI-PRMD conversion path, RehabExerAssess reference code, preprocessing, repetition detection, and model pipeline;
- helped define the canonical skeleton contract `[60, 12, 4]` in `collector/canonical.py`;
- corrected UI-PRMD coordinate conversion into the live MediaPipe-compatible convention, including Y-down semantics and normalized body-relative Z;
- added subject-safe UI-PRMD splitting, centralized label conventions, and regression tests;
- implemented a lightweight 12-joint binary form classifier, training script, saved-model inference wrapper, and live collector inference integration;
- added rule-based squat issue analysis on top of binary inference;
- redesigned the OpenCV HUD into a cleaner panel focused on status, repetitions, and last prediction;
- prepared GitHub project metadata such as this README, `.gitignore`, and CI.

## Repository Layout

```text
collector/
  main.py                    # Live camera app
  camera.py                  # OpenCV camera wrapper
  pose.py                    # MediaPipe Pose wrapper
  repetition.py              # Squat repetition state machine
  preprocessing.py           # Resampling + canonical normalization
  canonical.py               # Single source of truth for joints/channels/labels
  form_inference.py          # Model inference wrapper
  form_analysis.py           # Rule-based form issue diagnostics
  form_model.py              # 12-joint binary classifier
  train_form_classifier.py   # Offline UI-PRMD training pipeline
  uiprmd_adapter.py          # UI-PRMD -> canonical skeleton converter
  tests/                     # Regression tests
  RehabExerAssess-main/      # Reference research code used during development
```

Large local folders such as `UI-PRMD/`, generated datasets, virtual environments, and trained checkpoints are intentionally ignored by Git.

## Canonical Skeleton Format

The application uses one canonical representation:

```text
[T=60, V=12, C=4]
```

Joints:

```text
left_shoulder, right_shoulder,
left_hip, right_hip,
left_knee, right_knee,
left_ankle, right_ankle,
left_heel, right_heel,
left_foot_index, right_foot_index
```

Channels:

```text
x, y, z, visibility
```

Coordinates are body-centered and normalized. `y` follows image-space direction: positive downward. `z` is mid-hip-relative and normalized by torso scale.

## Setup

Use Python 3.11 on Windows.

```powershell
cd path\to\fintes_AI
python -m venv collector\venv
.\collector\venv\Scripts\python.exe -m pip install --upgrade pip
.\collector\venv\Scripts\python.exe -m pip install -r .\collector\requirements.txt
```

If you already have `collector\venv`, use that interpreter for all commands:

```powershell
.\collector\venv\Scripts\python.exe ...
```

## Run Live Collector

```powershell
cd path\to\fintes_AI
.\collector\venv\Scripts\python.exe .\collector\main.py
```

Controls:

```text
1-6    select collection class
SPACE  start / stop recording
Z      undo last saved sample
R      reset session counters
Q/ESC  quit
```

If a trained model exists at `collector/models/squat_binary/best_model.pt`, the app loads it automatically and shows the last repetition result in the HUD.

## Tests

```powershell
cd path\to\fintes_AI
.\collector\venv\Scripts\python.exe -m unittest discover -s collector\tests -v
```

The same test command runs in GitHub Actions on every push and pull request.

## UI-PRMD Data

The raw UI-PRMD dataset is not included in the repository.

Expected local layout:

```text
UI-PRMD/
  skl_whole/
    A01S01E02C01.skeleton
    ...
```

Convert raw UI-PRMD A01 Kinect skeletons into the reference `.txt` layout:

```powershell
.\collector\venv\Scripts\python.exe .\collector\RehabExerAssess-main\convert_uiprmd.py
```

Convert reference UI-PRMD files into canonical `[60, 12, 4]` samples:

```powershell
.\collector\venv\Scripts\python.exe .\collector\uiprmd_adapter.py
```

## Train Binary Squat Classifier

The current MVP trains a binary classifier:

```text
0 = incorrect
1 = correct
```

Training uses a deterministic subject-safe split. By default, subjects `08`, `09`, and `10` are held out for testing.

```powershell
.\collector\venv\Scripts\python.exe .\collector\train_form_classifier.py --epochs 30 --batch_size 16 --cpu
```

Generated artifacts are written under:

```text
collector/models/squat_binary/
```

That folder is ignored by Git.

## Notes For Contributors

- Do not commit raw datasets, generated `.npy` samples, virtual environments, or model checkpoints.
- Keep coordinate transformations explicit and centralized in `collector/canonical.py` and `collector/preprocessing.py`.
- Use the existing tests before opening a PR:

```powershell
.\collector\venv\Scripts\python.exe -m unittest discover -s collector\tests -v
```

## License

No top-level license has been selected yet. The nested `collector/RehabExerAssess-main` folder contains its own upstream license and README.
