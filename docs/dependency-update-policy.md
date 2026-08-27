# Dependency update policy

The checked-in `requirements*.txt` files are security-sensitive release inputs.
Only change them through a reviewed pull request that also changes the matching
`.in` source file and regenerates the lock with `make lock` in Python 3.11.

## Quarantine period

Use package versions published at least **seven days** ago; prefer fourteen days
for routine feature updates. Record the package name, chosen version, release
date, and audit result in the pull request description. Do not update a version
only by hand-editing a lock file.

An owner may approve an earlier version only for a security fix or a documented
compatibility incident. The pull request must link the advisory/incident and
state why the shorter quarantine is justified.

## Installation boundary

Development and CI install the generated lock with `--require-hashes` and
`--only-binary=:all`. No workflow may add a separate unpinned `pip install`
step. pip has no npm-style package lifecycle scripts; binary-only installation
is the project's control against executing an unexpected source build.

## Audit

Before merging any dependency/configuration change, run the **Dependency audit**
workflow manually against the committed lock. It also runs weekly using OSV.
Do not suppress a finding merely to obtain a green result: document it in
`README.md` and create a migration/remediation task. The current MediaPipe /
protobuf limitation is one such tracked blocker.
