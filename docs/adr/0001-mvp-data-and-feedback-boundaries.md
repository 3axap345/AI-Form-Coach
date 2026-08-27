# ADR 0001: MVP data and feedback boundaries

## Status

Accepted for the current prototype.

## Context

The project needs one representation shared by live MediaPipe collection, UI-PRMD
conversion, preprocessing, model inference, storage, and tests. It also uses external
datasets and rule-based feedback whose clinical calibration is not established.

## Decision

- Keep `collector/canonical.py` as the single skeleton contract: `[60, 12, 4]`, the
  documented landmark order, body-centred image-style coordinates, and visibility.
- Keep UI-PRMD/RehabExerAssess external to Git. Supported tools consume a user-provided
  converted-text root through `--source-root`; clean-checkout tests use fixtures.
- Keep the deployed form classifier bounded to the current binary `incorrect`/`correct`
  inference contract. It is a feedback aid, not a clinical classifier.
- Keep `collector/form_analysis.py` rule-based feedback as an explicitly uncalibrated
  engineering heuristic. Its intent, owner, and calibration TODOs live in
  `docs/form-analysis-intent.md`.

## Consequences

- Changing landmark order, sequence shape, labels, normalisation, classifier output,
  or feedback meaning needs conversion/inference tests and owner review.
- Training/evaluation is reproducible only after a lawful external dataset is supplied;
  runtime and CI remain camera- and dataset-independent.
- The product must retain its non-medical boundary until calibration evidence and an
  explicitly approved product decision say otherwise.
