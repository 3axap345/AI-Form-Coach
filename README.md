# AI Form Coach

Python 3.11 prototype for camera-based squat collection: MediaPipe pose extraction,
repetition detection, quality validation, form feedback, and canonical sample storage.
It is not a medical diagnostic tool.

## Quick start

```powershell
git clone <repository-url>
cd fintes_AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
make setup
make test
python .\collector\main.py
```

The live app needs a local camera. Use `1`–`6` to choose a class, `SPACE` to
record, `Z` to undo, `R` to reset counters, and `Q`/`ESC` to quit.

## Development and dependencies

```bash
make setup    # hash-locked development dependencies
make lint     # Ruff lint + format check
make test     # full unittest suite
make format   # apply Ruff fixes/formatting
make clean    # caches only
```

`requirements.in` is the runtime source; `requirements-dev.in` adds tooling.
Their `.txt` counterparts are generated exact-version SHA-256 locks. CI installs
`requirements-dev.txt` with `--require-hashes`. Edit `.in`, then run `make lock`;
never hand-edit a generated lock. `Makefile` honours `PYTHON`, e.g.
`make PYTHON=.venv/Scripts/python.exe test`.

MediaPipe is pinned to `0.10.14`, whose requirement is `protobuf>=4.25.3,<5`.
The explicit `protobuf==4.25.9` remains compatible and is newer than the
`<4.25.8` range affected by CVE-2025-4565.

## Model safety

Training writes `best_model.pt` as a weights-only `state_dict`; metadata is in
`model_metadata.json`. Inference uses `weights_only=True`, checks the exact key
set, and calls `strict=True`; it has no unsafe-pickle fallback. Set
`Config.form_model_sha256` to a 64-character checksum to verify deployed weights.

Models, generated samples, and datasets are excluded from Git. The current release
artefact is source application plus a separately provisioned model bundle. A
packaged/Docker release pipeline is intentionally a roadmap TODO.

## Configuration thresholds

Defaults are in `collector/config.py`. Ranges labelled implementation-only are
not clinical recommendations; calibrate them with recorded target-camera data.

| Parameter | Default, units, tuning |
| --- | --- |
| `standing_knee_angle` | `160°`, geometry 0–180. Higher requires straighter standing; lower accepts bend. Adjust for camera/subject bias. |
| `bottom_knee_angle` | `100°`, 0–180. Higher recognizes shallower bottoms; lower needs deeper flexion. Use labelled captures. |
| `hysteresis` | `8°`, non-negative. Higher filters noise but delays phases; lower is faster but can chatter. |
| `standing_confirm_frames` | `5` frames, positive integer. Higher rejects accidental starts but adds latency; lower starts sooner. Tune for FPS/noise. |
| `min_rep_duration_sec` | `0.4 s`. Higher rejects fast movements; lower admits more false triggers. |
| `max_rep_tracking_duration_sec` | `5.0 s`. FSM timeout only: higher keeps stalled tracking; lower resets earlier. |
| `max_saved_rep_duration_sec` | `5.0 s`. Quality gate only: higher saves slower reps; lower rejects them. |
| `smoothing_window` | `5` frames. Higher removes jitter but delays phases; lower responds sooner. |
| `min_avg_visibility` / `min_keypoint_visibility` | `0.6` / `0.4`, 0–1. Higher requires clearer poses; lower tolerates occlusion. |
| `max_missing_frame_ratio` | `0.15`, 0–1. Higher tolerates tracking loss; lower protects continuity. |
| `min_bbox_area_ratio` / `max_bbox_area_ratio` | `0.05` / `0.85`, area share 0–1. Raise min for a nearer subject; lower max for more framing room. |
| `form_analysis_bottom_window` / `form_analysis_smoothing_window` | `9` / `5` frames. Higher smooths diagnostics; lower is more local/noise-sensitive. |
| `form_analysis_min_visibility` | `0.4`, 0–1. Higher produces fewer, safer diagnostics. |
| `shallow_depth_knee_angle` | `115°`, 0–180. Higher flags more squats as shallow; lower flags only very shallow reps. |
| `excessive_forward_lean_angle` | `28°`, 0–90. Higher is more permissive; lower flags smaller leans. |
| `knee_valgus_ratio_drop` / `heel_instability_threshold` | `0.18` / `0.12`, non-negative implementation-only normalised values. Higher is less sensitive. |
| `max_form_issues_displayed` | `3`, positive integer. Higher shows more feedback; lower keeps the HUD focused. |

Do not change landmark order or the `[60, 12, 4]` format without conversion and
inference tests.

## Dataset preparation

Raw UI-PRMD and RehabExerAssess assets are absent from Git because of size and
upstream licensing. Download UI-PRMD yourself and place raw Kinect files at:

```text
UI-PRMD/skl_whole/A01S01E02C01.skeleton
```

The checked-in converter has workspace-specific paths. In this checkout:

```powershell
python .\collector\RehabExerAssess-main\convert_uiprmd.py
python .\collector\uiprmd_adapter.py
```

The first command produces
`collector/RehabExerAssess-main/data/UI-PRMD/{Correct,Incorrect}/Kinect/Skeletons/*.txt`.
The adapter writes `[60, 12, 4]` `.npy` files to
`collector/dataset/squat_uiprmd/{correct,incorrect}`. Training reads the converted
`.txt` source by default with a subject-safe split (08–10 held out). Tests use
synthetic fixtures, not local data; no maintained evaluation command exists.

Known limitation: make `convert_uiprmd.py` CLI-configurable before use from another
checkout. The bundled RehabExerAssess tree is reference code, not a live runtime
dependency.

## CI and merge policy

The single workflow runs `ruff check .`, `ruff format --check .`, and the complete
unittest suite in separate `lint` and `tests` jobs with pip caching. Errors are not
suppressed. Actions alone does not prove merges are blocked: configure Branch
Protection/Rulesets for `main` with required status checks `lint` and `tests`.

## Architecture

```text
camera (collector/camera.py)
  -> pose extraction (collector/pose.py)
  -> repetition detection (collector/repetition.py)
  -> quality/preprocessing (collector/quality.py, collector/preprocessing.py)
  -> ML/feedback (collector/form_inference.py, collector/form_analysis.py)
  -> storage (collector/storage.py)
```

`collector/live_flow.py` coordinates completed-repetition processing;
`collector/canonical.py` defines landmark and label contracts.
