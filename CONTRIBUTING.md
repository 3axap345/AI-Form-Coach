# Contributing to AI Form Coach

AI Form Coach is a desktop prototype for squat-form data collection and feedback.
Read [AGENTS.md](AGENTS.md) and the [README](README.md) before changing code.

## Before opening a pull request

1. Create an issue from `.github/ISSUE_TEMPLATE/engineering-task.md` for any
   multi-file, security, configuration, data-contract or user-visible change.
2. Preserve the canonical `[60, 12, 4]` input contract and landmark order unless
   the change includes migration tests for training and inference.
3. Do not commit datasets, generated samples, model weights, `.env` files or the
   external RehabExerAssess checkout.
4. Use the hash-locked development environment described in the README.

## Verification

Run the relevant commands before requesting review:

```bash
make lint
make test
make coverage        # business-logic changes
make typecheck       # typed-core changes
make build           # release/bundle changes
```

State commands run, their outcomes, hardware/data checks not performed and any
remaining risks in the pull request description. Changes to coaching claims,
clinical wording, thresholds or model/data contracts need project-owner approval.
