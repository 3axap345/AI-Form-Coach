# AI Form Coach development guide

## Project overview

AI Form Coach collects squat repetitions and converts them to canonical
`[60, 12, 4]` pose samples. The live collector is the primary product; UI-PRMD
tools support offline training.

## Architecture

```text
camera: collector/camera.py
  -> pose: collector/pose.py
  -> rep detection: collector/repetition.py
  -> quality: collector/quality.py
  -> preprocessing: collector/preprocessing.py
  -> model: collector/form_model.py + collector/form_inference.py
  -> feedback: collector/form_analysis.py
  -> storage: collector/storage.py
```

`collector/live_flow.py` owns completed-repetition processing.
`collector/canonical.py` is the landmark/channel/label contract.

## Commands

```bash
make setup
make lint
make test
make coverage
make typecheck
make build
make format
```

## Development rules

- Do not change training/inference formats or landmark order without tests.
- Do not commit datasets, generated samples, checkpoints, or virtual environments.
- Never use unsafe pickle/model loading; inference must remain weights-only.
- Document every new configuration parameter, including units and tuning effect.
- Add a regression test for a bugfix where practical.
- Before finishing, run `make lint` and `make test`; run `make coverage` after
  business-logic changes and `make typecheck` after changes to the typed core.
- Do not rewrite broad areas of the application without a concrete need and plan.

## AI-agent workflow

1. Find the existing implementation.
2. Identify affected tests and data contracts.
3. Make the smallest coherent change.
4. Run targeted tests.
5. Run the full suite.
6. Run coverage after business-logic changes and type checking after typed-core changes.
7. Update documentation for changed public behaviour/configuration.
