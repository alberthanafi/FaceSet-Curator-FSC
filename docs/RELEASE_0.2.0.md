<!-- Copyright © 2026 Hanafi Mohd Radi. All rights reserved. -->

# FaceSort 0.2.0 — by HMR Studio

Find the best. Sort the rest.

FaceSet Curator is now FaceSort. This release updates the Windows app title,
Help/About pages, reports, executable and installer branding, and adds the FSC
logo supplied by the author. It also includes searchable offline help and the
responsive setup screen with Start curation in a fixed top action bar.

## Download

Use FaceSort-Setup.exe to install or upgrade the Windows application.
FaceSort.zip contains the source. SHA256SUMS.txt provides download checksums.

The existing application ID, Python package/imports, legacy fsc commands, caches,
recovery journals, and log locations remain compatible. New Python installations
also provide facesort and facesort-gui commands. The GitHub repository URL is
unchanged.

## Verification and limitations

The release is checked with the 43-test suite, Python compilation, GUI startup
smoke checks, and actual CUDA inference in the build environment and packaged app.
No real source-image collections were used for release testing.

The installer is not digitally signed; Windows may display a SmartScreen warning.
The supplied logo retains its baked-in checkerboard border.

Copyright © 2026 Hanafi Mohd Radi. All rights reserved.
