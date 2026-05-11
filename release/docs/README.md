# release/docs/

**This directory is a vendored mirror.** The canonical source of every
file here lives under [`docs/release/`](../../docs/release/) in the
source repo; the vendored copies ship inside the published Kaggle and
HuggingFace bundles so an AI reviewer or offline reader can verify the
README's claims without network access.

## Do not edit files in this directory

Edits to files in `release/docs/` will be **silently discarded** the
next time anyone runs `python scripts/sync_release_docs.py`.  Edit the
file in `docs/release/` instead, then re-run the sync.

The sync script refuses to overwrite a destination whose mtime is
newer than the source's — so an accidental local edit will be caught
on the next sync invocation rather than silently destroyed.  Pass
`--force` to override that guard *only* if you've confirmed the local
edits are unwanted.

## What's vendored here (and why)

| File | Source | Why it ships in the bundle |
|---|---|---|
| `generation_method.md` | `docs/release/generation_method.md` | Full DGP description — what is / isn't modelled. |
| `channel_signal_audit.md` | `docs/release/channel_signal_audit.md` | Backing for the "channel signal is weak" claim. |
| `break_me_guide.md` | `docs/release/break_me_guide.md` | Nine adversarial patterns + detection recipes. |
| `feature_dictionary.md` | `docs/release/feature_dictionary.md` | Long-form per-feature documentation. |
| `v1_acceptance_gates_bands.yaml` | `docs/release/v1_acceptance_gates_bands.yaml` | Operational acceptance bands per gate. |
| `v2_decision_log.md` | `docs/release/v2_decision_log.md` | Accepted-for-v2 findings register. |
| `relational_table_schemas.csv` | (hand-authored here) | Per-column docs for all 9 relational tables.  Validated against live parquet schemas in `tests/scripts/test_build_release_metrics.py`. |

`relational_table_schemas.csv` is the one exception — it is authored
directly in this directory because it documents the *bundle*'s
parquet schemas, not anything in the leadforge package.
