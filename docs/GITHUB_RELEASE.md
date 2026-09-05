<!-- Copyright © 2026 Hanafi Mohd Radi. All rights reserved. -->

# GitHub release procedure

The Git repository contains source, documentation, tests, and packaging definitions only. Generated
executables, installers, caches, model files, and test output are excluded by `.gitignore`.

## Release assets

Attach binaries to a GitHub Release rather than committing them to the repository:

- `FaceSetCurator-Setup.exe` — full offline GPU installer.
- `FaceSet-Curator.zip` — optional exact source snapshot; GitHub also creates source archives automatically.
- SHA-256 checksums for every uploaded binary.

GitHub requires each individual Release asset to remain under 2 GiB. Always verify artifact size and
run both `FaceSetCurator.exe --doctor` and `FaceSetCurator.exe --gui-smoke` before publishing.

## Licensing

This is publicly visible proprietary source. Public visibility does not grant permission to copy,
modify, redistribute, sublicense, sell, or create derivative works. See the root `COPYRIGHT` file.

## Signing limitation

Current binaries contain publisher and copyright metadata but are not Authenticode-signed. Document
this clearly in release notes until a trusted code-signing certificate is configured.
