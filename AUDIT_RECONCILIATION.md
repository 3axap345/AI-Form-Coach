# AI Form Coach — Audit Reconciliation

## Current revision

- Branch: `codex/config-validation`
- Commit: `988b920c68a0f6fea6ebff0bd5f4dcb4652f9165`
- Audit baseline commit: `fae6dd2574833b0ad90a132c1a8e158d26a9bdf4`
- Audit files checked: 10 Markdown audits dated 2026-08-26.

The source audits describe the baseline commit, not this revision. Statuses below
are based on current code, configuration and attempted verification. A local
Python 3.14 interpreter is present but the project supports Python 3.11 only and
does not have the locked dependencies installed; therefore runtime verification is
reported separately instead of inferred from test-file presence.

## Executive summary

Old findings checked: 44 (overlapping findings from several audits are grouped
where they name the same underlying condition).

- FIXED: 31
- PARTIALLY FIXED: 5
- STILL PRESENT: 4
- REGRESSED: 0
- NOT APPLICABLE: 2
- CANNOT VERIFY: 2

The major improvements are real hash-locked dependencies, safe model loading with a
mandatory checksum at load time, CI lint/type/coverage/build jobs, a source bundle,
agent/task/decision documentation, and a 35-test organised suite covering live flow.
The two material remaining risks are the legacy MediaPipe/protobuf security
constraint and unverified provider-side branch protection/current CI success.

## Verification results

| Check | Result | Evidence |
|---|---|---|
| `make help`, `make lint`, `make test`, `make coverage`, `make typecheck`, `make build` | NOT AVAILABLE | GNU Make is not installed in this Windows checkout. |
| Ruff lint | PASS | `python -m ruff check .` |
| Ruff format | PASS | `python -m ruff format --check .` |
| Tests | CANNOT VERIFY | Direct discovery fails before collection because local Python 3.14 lacks `numpy`; 35 tests are statically discoverable. |
| Coverage | CANNOT VERIFY | Local interpreter lacks `coverage`; configuration has `fail_under = 63`. |
| `pip check` | PASS | `No broken requirements found` for the local interpreter; this does **not** validate the Python 3.11 lock. |
| Typecheck | CANNOT VERIFY | Local interpreter lacks `mypy`; CI config runs it under Python 3.11. |
| Build | PASS | `python scripts/build_source_bundle.py` created `dist/ai-form-coach-988b920.zip`; inspected archive had 33 files and no dataset/model/cache/Git paths. |
| `git diff --check` | PASS | No whitespace errors. |
| Working tree | PASS | Clean after this report is committed; build outputs are ignored. |

## Audit reconciliation matrix

| Audit | Finding | Original state | Current evidence | Status | Action |
|---|---|---|---|---|---|
| AI readiness | Agent entry point absent | No AGENTS/agent contract | `AGENTS.md` documents product, architecture, `[60,12,4]`, invariants, commands, data/model boundaries and handoff. | FIXED | No |
| AI readiness | Task intake / acceptance criteria absent | No issue template or durable scope | `.github/ISSUE_TEMPLATE/engineering-task.md` includes Goal, User impact, Context, affected files, Non-goals, acceptance, verification, handoff and risks. | FIXED | No |
| AI readiness | No verification/handoff route | README-only workflow | AGENTS requires targeted/full tests, coverage/typecheck where relevant and explicit handoff outcomes. | FIXED | No |
| AI readiness | Product context insufficient | No scenario/boundary map | `docs/product-context.md` and ADR 0001 define prototype/non-medical boundary, user scenario and feedback limits. | FIXED | Keep owner review for product claims. |
| AI readiness / cognitive debt | UI-PRMD reference path falsely looked tracked | README pointed to ignored `RehabExerAssess-main/convert_uiprmd.py` | README now says clean checkout has neither external project nor converter and lists supported `uiprmd_adapter.py` commands. | FIXED | No |
| Cognitive debt | Canonical representation could drift | Contract was present but only partially traced | `canonical.py` is the source of truth; preprocessing, adapter, model, training, inference and tests use `canonical_shape()` / canonical constants. Contract is 60 frames, 12 ordered joints, x/y/z/visibility. | FIXED | No |
| Cognitive debt | Threshold origin and form-rule intent undocumented | Values appeared as unsupported tuning | README threshold table plus `docs/form-analysis-intent.md` state units, defaults, effects, owner and calibration TODOs. | FIXED | Complete calibration evidence before clinical claims. |
| Cognitive debt | No executable intent for live capture/quality | FSM and quality gates untested | `test_repetition_quality.py` and `test_live_flow.py` cover confirmation, timeout, quality boundaries/rejection and happy path. | FIXED | Runtime result still needs Python 3.11 verification. |
| Cognitive debt | No durable decision record | README-only decisions | `docs/adr/0001-mvp-data-and-feedback-boundaries.md` records skeleton, external-data and prototype decisions. | FIXED | No |
| Config hygiene | `standing_confirm_frames` was inert | Config setting did not change FSM behaviour | `RepetitionDetector` increments `_standing_streak` and starts tracking only after the configured consecutive frames; regression tests cover boundary and accidental standing frames. | FIXED | No |
| Config hygiene | `max_rep_duration_sec` had two meanings | One setting affected tracking and quality | `max_rep_tracking_duration_sec` resets the FSM; `max_saved_rep_duration_sec` rejects saved reps in `quality.py`; independence is tested. | FIXED | No |
| Config hygiene | Values lacked validation | Invalid configuration could silently proceed | `Config.__post_init__` validates positive values, 0..1 values and min/max ordering; invalid ranges are tested. | FIXED | Add validation whenever a bounded setting is added. |
| Config hygiene | Environment-template suggestion | `.gitignore` implied `.env.example` although no environment variables were read | No runtime env configuration exists; ignoring `.env` is appropriate and a template would be misleading. | NOT APPLICABLE | Revisit only if env configuration is introduced. |
| Dead code | `_standing_streak` / `min_knee_angle` looked unused | Detector state did not influence observable FSM result | `_standing_streak` gates descent and `min_knee_angle` is emitted in `CompletedRep`/logging. | FIXED | No |
| Dead code | HUD diagnostics were invisible | Form issues/counters were calculated but not rendered | `main.py` passes `form_issues` to `HudState`; `draw_diagnostics` renders feedback and `test_hud_displays_runtime_diagnostics` asserts drawing calls. | FIXED | No |
| Dead code | `Camera._active_backend` write-only | Assigned in `_open()` but never read | Current search finds only assignment sites in `camera.py`. | STILL PRESENT | P2: remove it or expose it as diagnostic state. |
| Dead code | Canonical declarations without consumers | `SCALE_POLICY`, `UIPRMD_JOINTS`, `validate_joint_names` had no call sites | Current Python search finds declarations only. They do not validate a current path. | STILL PRESENT | P2: remove, or wire into adapter/metadata validation with tests. |
| Dead code / dependency hygiene | `collector/test_uiprmd.py` / undeclared `aeon` | Test-looking dataset probe could break discovery | File is removed; `rg aeon` finds only README historical note; discovery root contains no dataset probe. | FIXED | No |
| Dead code | Ignored RehabExerAssess tree looked like code surface | External local clone could be mistaken for project source | README/AGENTS/ADR explicitly classify it as external, excluded data tooling. | FIXED | No |
| Dependency hygiene | No source/lock separation or hashes | Direct pins only | `requirements.in` is runtime source; `requirements-dev.in` adds tools; generated `.txt` locks contain exact pins and SHA-256 hashes; `make lock` is documented. | FIXED | Regenerate only from `.in` on Python 3.11. |
| Dependency hygiene | CI could resolve/install untrusted packages | No hash enforcement and tooling was separate | All dependency-consuming jobs install `requirements-dev.txt` with `--require-hashes --only-binary=:all`; no later unpinned pip install exists. | FIXED | No |
| Dependency hygiene | Single dependency surface mixed runtime and tooling | One file served all purposes | Runtime and development inputs/locks are separated. | FIXED | Optional data tools remain external rather than runtime dependencies. |
| Dependency hygiene | `protobuf` vulnerable through MediaPipe | Baseline lacked a compatible remediation | Lock uses `mediapipe==0.10.14` and `protobuf==4.25.9`, constrained to `<5`; README records that the known audit finding requires a reviewed pose-backend migration. | STILL PRESENT | P1: migrate from legacy MediaPipe Solutions before claiming full dependency hygiene. |
| Dependency hygiene | No update quarantine policy | No documented age/review process | `docs/dependency-update-policy.md` requires seven days (prefer fourteen), release-date/audit evidence and owner-approved security exceptions. Enforcement is procedural, not machine-verifiable. | PARTIALLY FIXED | Make review/attestation enforceable if the project needs stronger supply-chain controls. |
| Vulnerabilities | Unsafe `torch.load` | Arbitrary checkpoint could execute pickle | Sole load site uses `weights_only=True`, Mapping/key-set validation and `load_state_dict(strict=True)`; no unsafe fallback exists. Negative test uses a benign executable pickle payload and asserts it is not run. | FIXED | No |
| Vulnerabilities | Integrity verification optional/dead | SHA function alone did not establish trust | `FormClassifierInference` refuses a missing/invalid/mismatched expected SHA; `main.py` disables inference when a provisioned model has no checksum; tests cover missing and wrong hashes. | FIXED | Release owner must provision independently reviewed hash. |
| Vulnerabilities | CI dependency audit absent | No recurring vulnerability scan | `dependency-audit.yml` runs OSV audit weekly/manual against hash lock. Known MediaPipe/protobuf issue remains intentionally visible, so audit is not a required status check. | PARTIALLY FIXED | Remove or redesign the alerting workflow if its known failure becomes noise; do not suppress the advisory. |
| CI/CD | Gate only ran tests | No lint, format, coverage, typecheck or build | `Tests` workflow contains mandatory-failing `lint`, `tests`, `coverage`, `typecheck`, and `build` jobs. | FIXED | No |
| CI/CD | No dependency cache | Reinstalled graph each run | `actions/setup-python@v6` uses pip cache keyed by `requirements-dev.txt` in all dependency jobs. | FIXED | No |
| CI/CD | Unclear failure semantics | Baseline had only one job | No `continue-on-error` or `|| true` in workflows; commands are direct and non-zero failures stop jobs. | FIXED | No |
| CI/CD | No artifact/build | No delivery artefact | `make build` / `build_source_bundle.py` creates deterministic tracked-source zip; CI uploads it. Inspected archive excludes Git/data/models/caches. | FIXED | No |
| CI/CD | Floating Actions/runner references | `@v4/@v5`, `*-latest` | Actions use current major tags (`checkout@v6`, `setup-python@v6`, `upload-artifact@v6`) and runners remain `windows-latest`/`ubuntu-latest`. This is reasonable major pinning, not immutable SHA pinning. | PARTIALLY FIXED | Pin to commit SHAs only if the threat model requires action supply-chain immutability. |
| CI/CD | Required checks / branch protection unconfirmed | Audit API returned 403 | README names required checks and prior GitHub UI evidence showed required labels, but `gh` is unavailable in this checkout and provider rules cannot be re-read. | CANNOT VERIFY | Repository owner: Settings → Rules → Protect main; verify exact required contexts. |
| CI/CD | No tagged release process | No release mechanism | Source-bundle artifact is defined; no tag/release publication is intentionally implemented for this desktop prototype. | PARTIALLY FIXED | P3: add tag release process when distributing application/model bundles. |
| Tests | No camera-independent user happy path | ML-only test suite | `test_live_flow_happy_path_saves_one_completed_repetition` exercises synthetic pose → detector → quality → preprocessing → fake classifier → temp storage metadata. | FIXED | Verify execution in Python 3.11 CI. |
| Tests | FSM/quality/storage branches untested | No live runtime coverage | Tests cover standing confirmation, tracking timeout, quality visibility/duration rejection, camera/pose wrappers, HUD, and direct storage save/close/undo. | FIXED | No |
| Tests | Suite mixed in one file | `test_pipeline.py` held all tests | Tests are split by data pipeline, inference, repetition/quality, live flow, interfaces, form analysis, storage and shared fixtures. | FIXED | No |
| Tests | No coverage command/gate | Measured baseline 32%, no enforced floor | `make coverage` invokes full unittest suite; `[tool.coverage.report]` displays missing lines and has `fail_under = 63`; CI coverage job runs it equivalently. | FIXED | Current measured percentage cannot be re-verified locally. |
| Tests | Current suite health unknown | Baseline had 12 passing tests | Static AST count is 35 tests. Current execution is blocked locally by missing Python 3.11 dependencies; no current provider run was available to inspect. | CANNOT VERIFY | Run CI or `make setup && make test && make coverage` under Python 3.11. |
| Tech health | No shared developer commands | README listed disparate commands | Makefile exposes help/setup/lint/test/coverage/typecheck/build/format/clean and README/AGENTS use them. | FIXED | Windows contributors need Make installed or direct documented equivalents. |
| Tech health | No pinned lint/formatter | Style was ungoverned | `pyproject.toml` configures Ruff; CI and Make use check/format modes. | FIXED | No |
| Tech health | Main loop combined responsibilities | Camera/UI loop owned processing and persistence | `LiveRepProcessor` extracts completed-repetition processing and has integration tests, but `main()` still orchestrates camera, keys, HUD and session state. | PARTIALLY FIXED | P2: split only if new runtime behavior makes it hard to test. |
| OSS readiness | Baseline 100 was only clean-fork non-applicability | No independent project surface was assessed | Current repository has substantial independent commits, README, issue template and documentation; old score is not transferable. | NOT APPLICABLE | Assess current project rather than fork baseline. |
| OSS readiness | Missing project governance files | Baseline did not assess them | No `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md` or `CODEOWNERS` is present; dataset/model licence terms are only partially described. | STILL PRESENT | P2/P3: decide project identity/licence and add governance docs if accepting external contributions. |

## AI readiness

The current agent surface is practical: AGENTS identifies the live collector as the
primary product, maps every layer to a file, protects the canonical contract and
sets verification/handoff expectations. The task template is usable, not merely a
placeholder. Dataset and model boundaries are explicit. The remaining limitation is
not missing instructions but unavailable local Python 3.11 tooling for verification.

## CI/CD

`Tests` triggers on `push` and `pull_request`; its Windows jobs use Python 3.11,
lock-file keyed pip cache, hash-locked binary-only installation, and direct failure
semantics. `lint` runs Ruff check and format check; `tests` runs unittest;
`coverage` runs coverage report; `typecheck` runs mypy; `build` uploads the source
zip. `Dependency audit` is scheduled/manual on Ubuntu and scans the lock via OSV.

Provider-side required checks cannot be proven from this checkout because `gh` is
not installed and no authenticated API is available. The local workflow describes
the required contexts; the owner must confirm them in GitHub Rulesets.

## Cognitive debt

Canonical format/order and the model/data boundary are consistently centralised.
Form-analysis intent is now recorded, including calibration TODOs rather than
invented clinical provenance. The remaining intentional uncertainty is calibration
against target camera data and upstream verification of UI-PRMD labels.

## Config hygiene

`standing_confirm_frames` changes the FSM transition, not just a value read by a
test. Tracking timeout and saved-rep duration govern different branches. Config
validation exists for the configured bounds. No new unreferenced Config field was
confirmed in the static pass.

## Dead code

Three small candidates remain: `_active_backend`, `SCALE_POLICY`, and
`UIPRMD_JOINTS`/`validate_joint_names`. They are not security or data-corruption
risks, so this reconciliation does not remove them merely for score improvement.

## Dependency hygiene

The lock is reproducible in principle (`make lock` under Python 3.11), exact and
hashed. The legacy MediaPipe Solutions dependency constrains protobuf below 5; the
repository accurately documents this blocker. A local vulnerability scan cannot be
run because the audit tool is not installed and current runtime dependencies are
absent. The scheduled/manual audit is not marked required because it would surface
the known unresolved finding.

## OSS readiness

The old formal 100/100 does not apply. This project now needs an explicit owner
decision: remain a project-specific repository, or add licence, contribution and
security-reporting surfaces suitable for outside contributors.

## Tech health

Commands, formatting and build artefacts have been materially improved. `clean`
only targets caches, bytecode and generated source bundles; its command does not
target datasets, model directories or user sample roots. The build script itself
uses tracked files and an explicit exclusion set.

## Tests

- Current static test count: 35
- Last locally observable configured baseline: coverage threshold 63%
- Current coverage: CANNOT VERIFY in this checkout (local `coverage` absent)
- Test structure: data pipeline, inference, repetition/quality, live flow,
  interfaces, form analysis, storage, and shared fixtures.

## Vulnerabilities

The original unsafe pickle and unhashed-install findings are fixed. The unresolved
dependency advisory remains a real medium supply-chain issue, not a suppressed
warning. No other `torch.load`, `joblib`, `yaml.load`, `eval`, `exec`, or
`shell=True` production path was found in the current source search.

## Priority backlog

### P0

None confirmed in the current checkout.

### P1

- Migrate legacy MediaPipe Solutions pose extraction so the project can use a
  protobuf release without the documented advisory.
- Run and preserve a current Python 3.11 CI/test/coverage/typecheck result before
  merging this branch; local environment cannot provide that evidence.

### P2

- Resolve the three confirmed dead-code candidates deliberately.
- Decide whether project identity requires LICENSE, CONTRIBUTING and SECURITY
  policy documents.
- Re-check ruleset required checks using authenticated GitHub access.

### P3

- Define tag-based release/versioning only when source+model distribution is a
  product need.
- Enforce the dependency-quarantine attestation mechanically only if procedural
  review is insufficient.

## Changes made during reconciliation

No production behavior was changed during this reconciliation. This report records
the evidence and outstanding work; the source bundle created for verification is
ignored by Git.

## Remaining issues

1. `mediapipe==0.10.14` requires `protobuf<5`, retaining the documented audit
   finding until pose extraction is migrated.
2. Current tests, coverage and mypy could not be executed locally because this
   machine lacks the supported locked Python 3.11 environment.
3. Provider-side required checks/ruleset cannot be independently verified here.
4. Four OSS governance documents are absent if this is intended for external
   contributors.

## Final assessment

| Area | Old score | Estimated current score | Status |
|---|---:|---:|---|
| AI readiness | 40 | 86 | mostly healthy |
| CI/CD | 48 | 79 | healthy with provider verification gap |
| Cognitive debt | 47 | 78 | improved; calibration remains |
| Config hygiene | 57 | 88 | healthy |
| Dead code | 64 | 78 | small confirmed cleanup backlog |
| Dependency hygiene | 43 | 69 | lock fixed; protobuf blocker remains |
| OSS readiness | 100* | 55 | now applicable; governance incomplete |
| Tech health | 74 | 84 | healthy |
| Tests | 54 | 80 | strong surface; current execution unverified locally |
| Vulnerabilities | 88 | 90 | unsafe load fixed; dependency advisory remains |

\* The old OSS 100/100 meant the clean-fork audit was not applicable, not that the
project had complete open-source governance.

## Evidence

- Agent contract: `AGENTS.md`; task template:
  `.github/ISSUE_TEMPLATE/engineering-task.md`.
- Canonical contract: `collector/canonical.py`; consumers: `collector/preprocessing.py`,
  `collector/uiprmd_adapter.py`, `collector/form_model.py`,
  `collector/form_inference.py`, `collector/train_form_classifier.py` and tests.
- FSM/quality: `collector/repetition.py`, `collector/quality.py`,
  `collector/tests/test_repetition_quality.py`.
- Live-flow and storage: `collector/live_flow.py`, `collector/storage.py`,
  `collector/tests/test_live_flow.py`, `collector/tests/test_storage.py`.
- Model boundary: `collector/form_inference.py`, `collector/main.py`,
  `collector/tests/test_inference.py`.
- Lock and CI: `requirements*.in`, `requirements*.txt`, `Makefile`,
  `.github/workflows/tests.yml`, `.github/workflows/dependency-audit.yml`.
- Intent/decision/data documentation: `README.md`, `docs/form-analysis-intent.md`,
  `docs/product-context.md`, `docs/adr/0001-mvp-data-and-feedback-boundaries.md`.
