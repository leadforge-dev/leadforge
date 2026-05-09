# PR 7.2 — Local Kaggle / HF preview-page design notes

Working notes for `scripts/preview_kaggle_page.py`,
`scripts/preview_hf_page.py`, their tests, and the committed
sample-rendered HTML used as the audit-artefact-sync gate. Captured
before implementation; kept short on purpose.

The PR's pedagogical role is the *staging gate* before PR 7.3: the
maintainer renders both platforms locally from the same artefacts the
publish PR will upload, clicks through them in a browser, and catches
styling / link / YAML-rendering issues before they hit cached
previews on the live page.

## Decisions

| # | Decision | Why |
|---|---|---|
| 1 | Two scripts, one per platform. Not a unified renderer. | Kaggle and HF have different inputs (`dataset-metadata.json` vs YAML-frontmatter README) and different page structures (schema/columns table vs configs dropdown). One file per platform keeps each renderer locally complete and the diff readable. |
| 2 | Server: stdlib `http.server.ThreadingHTTPServer` + `webbrowser.open()`. No Flask. | The pages are static HTML over a fixed file tree. A web framework would be a new dep with no benefit; the brief explicitly suggests stdlib. |
| 3 | Templates: f-string helpers, not Jinja2. | Layout is layout-stable; two pages don't justify a templating engine. f-string helpers keep the renderer in one file and free of a new dep. |
| 4 | Markdown→HTML via `markdown-it-py` (added to `[publish]` extra alongside `datasets` / `kaggle`). | Faithfulness is the goal — Kaggle and HF both render the README body as Markdown, hand-rolling a renderer for tables / fenced code / footnotes is brittle. `markdown-it-py` is MIT, pure-Python, CommonMark+GFM. The `[publish]` extra is the right home: this is a publish-pipeline tool, mirrors the PR 5.1 / 5.2 gating posture. Missing dep raises a clean `ImportError` that points at `pip install -e ".[publish]"`. |
| 5 | Output dir: `release/_preview/<platform>/` (gitignored). | Mirrors `release/_release_quality/` convention. The committed audit-sync samples live at `release/_preview_committed/{kaggle,huggingface_public,huggingface_instructor}.html` so they don't collide with runtime output. |
| 6 | Cover image served from the preview tree (copied in, not referenced). | Both platforms inline-display the cover image; serving it under the preview root means the rendered HTML's `<img src="dataset-cover-image.png">` works without absolute paths. The committed sample HTML uses the same relative reference — no path drift between the sample and what the local server emits. |
| 7 | HF `--variant=public|instructor` reads either `release/huggingface/README.md` or `release/huggingface-instructor/README.md`. Different YAML, different file tree, different name. Kaggle has no instructor variant (Kaggle ships public only). | Matches the publish reality (HF gets a separate instructor companion repo per PR 5.2; Kaggle does not). |
| 8 | CLI mirrors `validate_release_candidate.py` / `run_llm_critique.py`: free-function `parse_args`, frozen `Config`, `run_preview(config) -> Outcome`, `main(argv) -> int`. Exit codes 0 success / 2 pre-flight error. Flags: `--release-dir`, `--port` (8765 Kaggle / 8766 HF), `--out-dir`, `--variant` (HF only), `--open-browser`, `--no-serve`. | Maintainer muscle memory + small surface. `--no-serve` is the CI / inspection mode (build HTML, exit 0). `--open-browser` pops a tab on startup. |
| 9 | Audit-artifact-sync. The renderer is pure: `(metadata.json | README + YAML, cover image filename) -> HTML`. No `now()`, no random. Committed HTML at `release/_preview_committed/*.html` must equal a fresh regeneration byte-for-byte. Same pattern as PR 4.1 / 5.1 / 5.2 / 7.1. | Determinism is the gate against silent drift. The committed HTML doubles as a human-inspectable sample for reviewers who don't want to run the script. |
| 10 | Test posture: in-process. No live HTTP. Each test renders the page once via `render_kaggle_html()` / `render_hf_html()` and asserts against the rendered string with substring + regex. No BeautifulSoup dep (avoidable for the assertion bar we need). The four roadmap-mandated checks: required field labels appear; every Markdown link in the source resolves to a non-404 URL pattern; every config block (HF) round-trips; the Kaggle schema table lists every CSV / parquet column from `resources[].schema.fields`. | Per the brief — no live HTTP, no new test deps unless necessary. Substring assertions on deterministic rendered HTML give the same coverage with less surface. |

## Link-resolution rule (test pin)

Every Markdown link `](URL)` in the README body the renderer ingests
must satisfy ONE of:

1. Absolute `https://github.com/leadforge-dev/leadforge/...` URL (the
   rewrite output of `_release_common.py::rewrite_release_links()`).
2. External absolute URL on a known-OK domain (`https://huggingface.co`,
   `https://github.com/leadforge-dev/leadforge`, footnote anchors).
3. Relative path that resolves to a file under the upload tree
   (e.g. `LICENSE` → `release/<platform>/LICENSE`).

A `](../foo)` link or a `](validation/...)` link in the rendered
HTML is a regression — those are exactly what the platform packagers'
rewrite is supposed to canonicalise away. The test fires loud the
moment the rewrite stops doing its job for the upstream artefact the
preview renders.

## What this PR does not touch

- `BUNDLE_SCHEMA_VERSION` stays at 5.
- `release/validation/validation_report.{json,md}` does not regenerate
  (revert any timestamp drift before commit).
- PR 7.3 (publish + tag) is a separate PR; the runbook there will cite
  the two preview commands as a required pre-flight step.
- No change to the platform packagers (`scripts/package_{kaggle,hf}_release.py`)
  or `_release_common.py`. The preview reads what the packagers wrote.
- Live Kaggle / HF API calls — pure local rendering only.
- Pixel-perfect cloning of the live pages. The bar is "a maintainer
  clicking through it would notice the same broken link, malformed
  YAML, or missing config that they'd notice on the live page".
