# De-identification of deposited files

The files in this repository were de-identified before deposit. Analytical
values are unchanged; nothing that enters a model was altered.

## What was removed

**Columns.** Any column whose name indicates a file path, a filename, a video
or clip reference, free-text notes or comments, a reviewer or operator name, or
an absolute calendar date was dropped in full. Each dropped column is named in
the preparation log.

**Residual values.** Remaining cells were scanned for local file paths,
`OneDrive - <institution>` strings, user directories, email addresses, video
filenames and ISO dates. A matching cell was replaced entirely with
`[REDACTED]` rather than patched, because a path can carry a person's name in a
directory component that a pattern may not reach.

**Scripts.** Hard-coded absolute paths in deposited `.py` files were replaced
with `<SET_YOUR_ANALYSIS_ROOT>`. The scripts document the analysis; set a root
before running them.

Participant codes (P1-P30) are retained. They are pseudonyms, they carry no
identifying content on their own, and they are required to link records across
the three SIMETRIC repositories.

## SHA256 manifests

`08_SIMETRIC_grip_final_freeze_SHA256_manifest_v1_8_2.csv` and the other
`*_hashes.csv` manifests were computed on the files **as frozen internally**,
before de-identification. De-identification changes the bytes of those files,
so the recorded digests will not verify against the copies published here.

The manifests are retained as a record of the internal freeze and of file
membership, not as a checksum for the deposited copies. Row counts, column
membership other than the dropped columns, and every analytical value are
unchanged from the frozen release.
