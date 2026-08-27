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

For intended users, supported coaching claims, terminology, and UX limits, see
[docs/product-context.md](docs/product-context.md).

The MVP's skeleton/data/feedback decisions are recorded in
[ADR 0001](docs/adr/0001-mvp-data-and-feedback-boundaries.md). The owner, intent, and
calibration TODOs for form-analysis and quality thresholds are in
[docs/form-analysis-intent.md](docs/form-analysis-intent.md).

## Development and dependencies

```bash
make setup    # hash-locked development dependencies
make lint     # Ruff lint + format check
make test     # full unittest suite
make coverage # production-code coverage, including missing lines
make typecheck # mypy baseline for the typed core pipeline
make build    # deterministic source bundle in dist/
make format   # apply Ruff fixes/formatting
make clean    # caches only
```

`requirements.in` is the runtime source; `requirements-dev.in` adds tooling
(`pip-tools`, Ruff, coverage, and mypy).
Their `.txt` counterparts are generated exact-version SHA-256 locks. CI installs
`requirements-dev.txt` with `--require-hashes --only-binary=:all`: the lock prevents
tampering and unpinned versions, and the binary-only policy prevents an unexpected
source build. Edit `.in`, then run `make lock`; never hand-edit a generated lock.
`Makefile` honours `PYTHON`, e.g.
`make PYTHON=.venv/Scripts/python.exe test`.

`pyproject.toml` is the source of truth for the supported Python range; CI currently
tests Python 3.11. Configuration defaults and their validation live in
`collector/config.py`; the threshold table below explains their safe tuning context.

MediaPipe is pinned to `0.10.14`, whose legacy `mp.solutions.pose` API is used by
`collector/pose.py`; it requires `protobuf>=4.25.3,<5`. `pip-audit` currently
reports `PYSEC-2026-1805` for the compatible `protobuf==4.25.9`, whose fix requires
protobuf 5.29.6 or newer. MediaPipe releases that allow that upgrade do not retain
the legacy API on the supported Windows path. Do not suppress this finding: closing
it requires a separately reviewed migration of pose extraction to MediaPipe Tasks or
another supported pose backend.

## Model safety

Training writes `best_model.pt` as a weights-only `state_dict`; metadata is in
`model_metadata.json`. Inference uses `weights_only=True`, checks the exact key
set, and calls `strict=True`; it has no unsafe-pickle fallback. Set
`Config.form_model_sha256` to a 64-character checksum to verify deployed weights.

Models, generated samples, and datasets are excluded from Git. The current release
artefact is a deterministic, Git-tracked source bundle plus a separately provisioned
model bundle: `make build` writes `dist/ai-form-coach-<commit>.zip`. The CI `build`
job verifies and uploads that source artifact. A packaged/Docker release pipeline is
intentionally a roadmap TODO.

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

UI-PRMD data and `RehabExerAssess-main` are external/local inputs: neither is in
Git, and both are intentionally ignored due to upstream licensing and size. A clean
checkout does **not** contain `collector/RehabExerAssess-main` or a converter for raw
`.skeleton` files.

Obtain UI-PRMD and, if required, perform raw `.skeleton` conversion with your own
lawful external RehabExerAssess checkout or another verified converter. This project
expects the resulting converted-text tree to have this layout:

```text
<converted-ui-prmd>/
  Correct/Kinect/Skeletons/A01S01E02C01.txt
  Incorrect/Kinect/Skeletons/A01S01E02C02.txt
```

From a clean checkout, these are the supported commands:

```powershell
python .\collector\uiprmd_adapter.py --source-root <converted-ui-prmd>
python .\collector\train_form_classifier.py --source-root <converted-ui-prmd>
```

The adapter writes `[60, 12, 4]` `.npy` files under
`collector/dataset/squat_uiprmd/{correct,incorrect}`. Training reads converted `.txt`
files directly and makes a subject-safe split (08–10 held out); it does not consume
the adapter output. Tests use synthetic fixtures, not local datasets, and no
maintained evaluation command exists.

To bypass all external-data work, use `make test`: it is a deterministic headless
smoke suite using synthetic landmarks and mocked camera/pose components. The live
application still requires a local camera; training requires the external converted
text tree.

The old one-off `aeon` dataset experiment is intentionally not part of the supported
workflow and has been removed; no application or test dependency on `aeon` remains.

## CI and merge policy

The single workflow has separate `lint`, `tests`, `coverage`, `typecheck`, and
`build` jobs. The first four install only `requirements-dev.txt` using
`--require-hashes --only-binary=:all`; no CI step installs an unpinned package or
builds an unpinned source distribution. Coverage excludes test files, reports
missing lines, and fails below the current production baseline of 63%. `build`
creates and uploads the source bundle. Errors are not suppressed.

The active `Protect main` ruleset targets `main`, requires `lint`, `tests`,
`coverage`, `typecheck`, and `build`, and blocks deletion and force-pushes. It does
not currently require a pull request/review or an up-to-date branch before merging;
enable those options in the ruleset if that workflow is desired.

## Verification contract

`make test` is the baseline proof for every change and works without a camera, model,
or dataset. `make lint` is required before handoff; use `make coverage` for
business-logic changes and `make typecheck` when modifying the typed core pipeline.
`make format` changes source files and is optional when checks already pass.

For a headless proof, attach the commands and their outcomes from these targets. A
camera smoke test is manual (`python .\collector\main.py`) and is not required when
hardware is unavailable; state that limitation in the handoff. Dataset conversion and
training are likewise optional external checks unless the change affects that workflow.

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
