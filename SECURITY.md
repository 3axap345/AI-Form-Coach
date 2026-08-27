# Security policy

## Supported version

Security fixes are made on the default development branch while the project remains
a prototype. No packaged release line is currently supported.

## Reporting a vulnerability

Use GitHub's private vulnerability-reporting feature for this repository when it is
enabled. Do not include exploit details, secrets, model artefacts or personal video
data in public issues.

If private reporting is unavailable, contact the repository owner through the
maintainer channel listed on the repository profile and include: affected commit,
reproduction steps, impact and suggested mitigation.

## Scope notes

Model files are untrusted until their independently reviewed SHA-256 is supplied.
The repository has a documented legacy MediaPipe/protobuf dependency constraint;
reports about a new or changed advisory should include the package, version and
advisory identifier.
