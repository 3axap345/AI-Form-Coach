# AI Form Coach development guide

## Project overview

AI Form Coach collects squat repetitions and converts them to canonical
`[60, 12, 4]` pose samples. The live collector is the primary product; UI-PRMD
tools support offline training.

Read [README.md](README.md) first: **Quick start** describes the live app,
**Configuration thresholds** explains safe tuning, **Dataset preparation** defines
the external UI-PRMD boundary, and **CI and merge policy** defines required checks.

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
Use `collector/preprocessing.py` for normalisation and `collector/config.py` for
validated runtime thresholds; do not create a parallel landmark representation.

## Commands

```bash
make help
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
- Do not commit `.env` files, local models, cached artefacts, or the external
  `collector/RehabExerAssess-main/` checkout.
- Never use unsafe pickle/model loading; inference must remain weights-only.
- Treat a model SHA-256 as a required deployment trust anchor: never load a
  provisioned model without its independently reviewed checksum.
- For dependency changes, follow `docs/dependency-update-policy.md`; regenerate
  locks, retain hashes, and run the manual Dependency audit workflow before merge.
- Do not alter medical/non-medical claims, coaching-rule intent, or clinical wording
  without explicit project-owner approval and an update to `docs/form-analysis-intent.md`.
- Document every new configuration parameter, including units and tuning effect.
- Extend `Config` validation and its regression tests when adding a bounded setting.
- Add a regression test for a bugfix where practical.
- Before finishing, run `make lint` and `make test`; run `make coverage` after
  business-logic changes and `make typecheck` after changes to the typed core.
- Do not rewrite broad areas of the application without a concrete need and plan.

## External assumptions and handoff

- Camera, deployed model weights, UI-PRMD data, and RehabExerAssess are local/external
  inputs; a clean checkout must still support headless tests without them.
- Use `.github/ISSUE_TEMPLATE/engineering-task.md` for multi-file, security,
  configuration, CI, data-contract, or user-visible changes.
- At handoff, state the goal, affected files/contracts, commands run and outcomes,
  any unrun hardware/data checks, and remaining risks or follow-ups.

## AI-agent workflow

1. Find the existing implementation.
2. Identify affected tests and data contracts.
3. Make the smallest coherent change.
4. Run targeted tests.
5. Run the full suite.
6. Run coverage after business-logic changes and type checking after typed-core changes.
7. Update documentation for changed public behaviour/configuration.
