"""Tests for ``scripts/package_hf_release.py``.

Locks the Phase 5 Hugging Face packaging contract:

* every YAML field surfaced in chatgpt v2 §20 + G12.1 (pretty_name,
  license, language, task_categories, size_categories, tags, configs
  with data_files)
* G12.2 — exactly one config has ``default: true``
* G12.3 — local ``load_dataset()`` succeeds for every config and
  returns the expected row counts (gated on the optional ``datasets``
  package; ``pytest.importorskip`` if absent)
* G12.4 — instructor companion variant packages independently and
  loads via ``load_dataset()``
* the README link-rewriting that lets the published dataset card on
  HF keep working ``../`` links (rewritten to GitHub blob URLs) and a
  directory diagram that reflects the upload layout, plus a guard
  that the source ``SOURCE_TREE_BLOCK`` (in ``_release_common``) is still present verbatim
  in the README (silent-failure trap; mirrors PR 5.1's KAGGLE block)
* the assembled upload tree resolves every declared resource path
* the safety net that refuses to assemble into ``cwd`` /
  ``release_dir`` / its parent
* byte-equality between the committed
  ``release/huggingface/README.md`` and a fresh regeneration
  (audit-artifact-sync pattern from PR 4.1 / PR 5.1)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "package_hf_release.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("package_hf_release", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
packager = importlib.util.module_from_spec(_spec)
sys.modules["package_hf_release"] = packager
_spec.loader.exec_module(packager)


_RELEASE_DIR = _REPO_ROOT / "release"
_RELEASE_BUNDLES_PRESENT = (_RELEASE_DIR / "intro" / "manifest.json").exists()
_INSTRUCTOR_BUNDLE_PRESENT = (_RELEASE_DIR / "intermediate_instructor" / "manifest.json").exists()
_COMMITTED_README = _REPO_ROOT / "release" / "huggingface" / "README.md"
_COMMITTED_INSTRUCTOR_README = _REPO_ROOT / "release" / "huggingface-instructor" / "README.md"
_COMMITTED_COVER = _REPO_ROOT / "release" / "dataset-cover-image.png"


# Canonical task-split row counts per public tier (3500 / 750 / 750)
# — pinned so the load_dataset() smoke test fails loud rather than
# silently accept an empty / truncated parquet.
_EXPECTED_ROWS = {"train": 3500, "validation": 750, "test": 750}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_card() -> packager.HuggingFaceCard:
    """A minimum-viable ``HuggingFaceCard`` that should validate cleanly."""

    return packager.HuggingFaceCard(
        pretty_name=packager.DEFAULT_PRETTY_NAME,
        license=packager.DEFAULT_LICENSE,
        language=(packager.DEFAULT_LANGUAGE,),
        task_categories=(packager.HF_TASK_CATEGORY,),
        size_categories=(packager.HF_SIZE_BUCKET_5K,),
        tags=packager.DEFAULT_TAGS,
        configs=(
            packager.ConfigEntry(
                config_name="intermediate",
                data_files=(
                    packager.DataFileEntry(
                        split="train",
                        path="intermediate/tasks/converted_within_90_days/train.parquet",
                    ),
                ),
                default=True,
            ),
        ),
        body="# LeadForge\n",
    )


# ---------------------------------------------------------------------------
# Field-constraint validation (G12.1)
# ---------------------------------------------------------------------------


def test_validate_card_accepts_minimal_canonical_card() -> None:
    assert packager.validate_card(_minimal_card()) == []


def test_validate_card_reports_every_field_violation() -> None:
    """One bad card triggers every field check at once."""

    bad = packager.HuggingFaceCard(
        pretty_name="",
        license="apache-2.0",  # not 'mit'
        language=(),
        task_categories=("image-classification",),  # missing tabular-classification
        size_categories=(),
        tags=(),
        configs=(),
        body="",
    )
    errors = packager.validate_card(bad)
    fields = {e.field for e in errors}
    assert "pretty_name" in fields
    assert "license" in fields
    assert "language" in fields
    assert "task_categories" in fields
    assert "size_categories" in fields
    assert "tags" in fields
    assert "configs" in fields


def test_validate_card_requires_exactly_one_default() -> None:
    """G12.2 — exactly one config carries ``default: true``."""

    base = _minimal_card()

    # Zero defaults.
    zero = packager.HuggingFaceCard(
        **{
            **base.__dict__,
            "configs": (
                packager.ConfigEntry(
                    config_name="intro",
                    data_files=(packager.DataFileEntry(split="train", path="intro/x.parquet"),),
                    default=False,
                ),
            ),
        }
    )
    errors = packager.validate_card(zero)
    assert any("default: true" in e.message for e in errors)

    # Two defaults.
    two = packager.HuggingFaceCard(
        **{
            **base.__dict__,
            "configs": (
                packager.ConfigEntry(
                    config_name="intro",
                    data_files=(packager.DataFileEntry(split="train", path="intro/x.parquet"),),
                    default=True,
                ),
                packager.ConfigEntry(
                    config_name="intermediate",
                    data_files=(
                        packager.DataFileEntry(split="train", path="intermediate/x.parquet"),
                    ),
                    default=True,
                ),
            ),
        }
    )
    errors = packager.validate_card(two)
    assert any("default: true" in e.message for e in errors)


def test_validate_card_flags_data_files_without_split_or_path() -> None:
    """Each ``data_files[]`` entry must declare both split and path."""

    base = _minimal_card()
    broken = packager.ConfigEntry(
        config_name="intermediate",
        data_files=(packager.DataFileEntry(split="", path="intermediate/x.parquet"),),
        default=True,
    )
    bad = packager.HuggingFaceCard(**{**base.__dict__, "configs": (broken,)})
    errors = packager.validate_card(bad)
    assert any("split and path" in e.message for e in errors)


# ---------------------------------------------------------------------------
# YAML rendering — round-trip parse + content invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pretty_name",
    [
        packager.DEFAULT_PRETTY_NAME,
        # Em-dash flavor — matches the instructor variant; locks down
        # PyYAML round-tripping for non-ASCII content (regression
        # guard for the PR 5.2 self-review note about the hand-rolled
        # renderer's incomplete coverage).
        packager.DEFAULT_INSTRUCTOR_PRETTY_NAME,
    ],
)
def test_render_yaml_frontmatter_round_trips_through_pyyaml(pretty_name: str) -> None:
    """The rendered YAML must parse cleanly via ``yaml.safe_load``.

    HF parses dataset-card frontmatter with PyYAML; if our renderer
    drifts from valid YAML, the dataset card silently drops its
    metadata on the HF page.
    """

    base = _minimal_card()
    card = packager.HuggingFaceCard(**{**base.__dict__, "pretty_name": pretty_name})
    yaml_text = packager.render_yaml_frontmatter(card)
    # ``render_yaml_frontmatter`` includes the leading and trailing
    # ``---`` markers.  Strip them before feeding to safe_load.
    inner = yaml_text.strip().strip("-").strip()
    parsed = yaml.safe_load(inner)
    assert parsed["pretty_name"] == pretty_name
    assert parsed["license"] == "mit"
    assert parsed["language"] == ["en"]
    assert parsed["task_categories"] == [packager.HF_TASK_CATEGORY]
    assert parsed["size_categories"] == [packager.HF_SIZE_BUCKET_5K]
    assert parsed["tags"] == sorted(card.tags)
    assert len(parsed["configs"]) == 1
    assert parsed["configs"][0]["default"] is True


def test_render_yaml_tags_sorted_at_render_time() -> None:
    """Tags are sorted in the rendered YAML regardless of dataclass order."""

    base = _minimal_card()
    shuffled = packager.HuggingFaceCard(
        **{**base.__dict__, "tags": ("zebra", "alpha", "mango")},
    )
    rendered = packager.render_yaml_frontmatter(shuffled)
    parsed = yaml.safe_load(rendered.strip().strip("-").strip())
    assert parsed["tags"] == ["alpha", "mango", "zebra"]


# ---------------------------------------------------------------------------
# README rewriting + content
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_hf_public_readme_text_rewrites_links_and_tree_diagram() -> None:
    readme = (_RELEASE_DIR / "README.md").read_text(encoding="utf-8")
    rewritten = packager._hf_public_readme_text(readme)

    # Source-repo tree → upload tree.
    assert "intermediate_instructor/" not in rewritten
    assert "notebooks/01_baseline_lead_scoring.ipynb" not in rewritten
    assert "README.md                         # this file (HF dataset card)" in rewritten

    # Relative ../ links rewritten to GitHub blob URLs.
    assert "](../" not in rewritten
    assert packager.GITHUB_BLOB_BASE in rewritten

    # The validation-report link (which lives under release/, not under
    # the upload dir) must point at GitHub.
    assert "](validation/validation_report.md)" not in rewritten
    assert f"]({packager.GITHUB_BLOB_BASE}/release/validation/validation_report.md)" in rewritten


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_source_tree_block_is_present_in_release_readme() -> None:
    """Silent-failure guard.

    ``_hf_public_readme_text`` substitutes ``SOURCE_TREE_BLOCK`` →
    ``HF_UPLOAD_TREE_BLOCK`` via plain string replace.  If anyone
    tweaks the README's tree diagram by even one whitespace
    character, the substitution silently no-ops and the published HF
    dataset card carries the source-repo tree.  This guard fires
    loudly the moment the constants drift apart.
    """

    readme = (_RELEASE_DIR / "README.md").read_text(encoding="utf-8")
    assert packager.SOURCE_TREE_BLOCK in readme, (
        "scripts/_release_common.py SOURCE_TREE_BLOCK no longer matches "
        "the tree diagram in release/README.md — reconcile the two before "
        "the next HF README regeneration."
    )


def test_validate_readme_substitution_flags_drift(tmp_path: Path) -> None:
    """``validate_readme_substitution`` (now in ``_release_common``) is
    wired into the run-time validator, not just the static guard above."""

    fake_release = tmp_path / "release"
    fake_release.mkdir()
    (fake_release / "README.md").write_text("# Some unrelated README\n", encoding="utf-8")
    errors = packager.validate_readme_substitution(fake_release, packager_name="HF")
    assert errors
    assert errors[0].field == "release/README.md"
    assert "SOURCE_TREE_BLOCK" in errors[0].message

    if _RELEASE_BUNDLES_PRESENT:
        # Sanity: the real release README does NOT trigger the validator.
        assert packager.validate_readme_substitution(_RELEASE_DIR, packager_name="HF") == []


# ---------------------------------------------------------------------------
# Upload-dir assembly safety + idempotence
# ---------------------------------------------------------------------------


def test_assemble_upload_dir_rejects_unsafe_huggingface_dir(tmp_path: Path) -> None:
    """Refuse to assemble into the release dir or its parent."""

    fake_release = tmp_path / "release"
    fake_release.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        packager.assemble_upload_dir(fake_release, fake_release, rendered_readme="")
    with pytest.raises(ValueError, match="unsafe"):
        packager.assemble_upload_dir(fake_release, fake_release.parent, rendered_readme="")


def test_assemble_upload_dir_rejects_huggingface_dir_equal_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse to assemble into the current working directory.

    A user passing ``--huggingface-dir .`` would otherwise rmtree-then-
    recopy arbitrary cwd contents.  Same safety case as the Kaggle
    packager (PR 5.1 fix-up).
    """

    fake_release = tmp_path / "release"
    fake_release.mkdir()
    cwd = tmp_path / "workdir"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    with pytest.raises(ValueError, match="unsafe"):
        packager.assemble_upload_dir(fake_release, cwd, rendered_readme="")


def test_assemble_upload_dir_rejects_descendant_of_release_dir(tmp_path: Path) -> None:
    """Refuse to assemble into a strict descendant of ``release_dir``.

    Per PR 5.2 self-review #3: ``--huggingface-dir release/intro``
    would otherwise rmtree the intro tier bundle.  Only direct
    children of ``release_dir`` are allowed (the canonical
    ``release/huggingface``, ``release/huggingface-instructor``,
    ``release/kaggle`` shapes).
    """

    fake_release = tmp_path / "release"
    (fake_release / "intro" / "tables").mkdir(parents=True)
    deep = fake_release / "intro" / "tables"  # 2 levels under release_dir
    with pytest.raises(ValueError, match="unsafe"):
        packager.assemble_upload_dir(fake_release, deep, rendered_readme="")


def test_assemble_upload_dir_allows_canonical_child(tmp_path: Path) -> None:
    """Direct-child of release_dir IS the canonical safe location.

    ``release/huggingface`` is allowed; only deeper nesting trips the
    descendant guard.
    """

    fake_release = tmp_path / "release"
    fake_release.mkdir()
    safe_child = fake_release / "huggingface"
    # Should not raise.  The function may still fail later because the
    # source bundles aren't present, but the safety guard must let it
    # through.
    try:
        packager.assemble_upload_dir(fake_release, safe_child, rendered_readme="")
    except ValueError as exc:
        if "unsafe" in str(exc):
            raise AssertionError(f"safety guard rejected canonical child: {exc}") from exc
    except FileNotFoundError:
        # Expected — fake_release has no tier bundles, so the copytree
        # call fails after the safety check.  That's the right shape.
        pass


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_assembled_upload_dir_resolves_every_declared_data_file(tmp_path: Path) -> None:
    """Every ``configs[*].data_files[*].path`` declared in the YAML
    must resolve to a real file (not a symlink, not a missing path)
    under the assembled upload directory.  HF's ``datasets`` library
    walks the directory at upload time; a declared path that doesn't
    materialise is a silent load-time failure.
    """

    upload_dir = tmp_path / "huggingface"
    outcome = packager.run_packager(
        _RELEASE_DIR,
        huggingface_dir=upload_dir,
        cover_image=_COMMITTED_COVER,
    )

    for config in outcome.card.configs:
        for df in config.data_files:
            target = upload_dir / df.path
            assert target.is_file(), f"declared data_file missing from upload tree: {df.path}"
            assert not target.is_symlink(), (
                f"declared data_file is a symlink, not a real file: {df.path} — "
                f"datasets library may skip symlinked entries on upload"
            )

    # Top-level required artefacts.
    assert (upload_dir / "README.md").is_file()
    assert (upload_dir / _COMMITTED_COVER.name).is_file()
    assert (upload_dir / "LICENSE").is_file()


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_run_packager_idempotent_against_existing_tree(tmp_path: Path) -> None:
    """Re-running the assembly over an already-populated upload tree
    succeeds — both passes call the same copy helpers."""

    upload_dir = tmp_path / "huggingface"
    packager.run_packager(_RELEASE_DIR, huggingface_dir=upload_dir, cover_image=_COMMITTED_COVER)
    outcome = packager.run_packager(
        _RELEASE_DIR, huggingface_dir=upload_dir, cover_image=_COMMITTED_COVER
    )
    assert outcome.errors == ()
    for config in outcome.card.configs:
        for df in config.data_files:
            assert (upload_dir / df.path).is_file()


# ---------------------------------------------------------------------------
# CLI driver — error paths
# ---------------------------------------------------------------------------


def test_main_reports_missing_release_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = packager.main(
        [
            "--release-dir",
            str(tmp_path / "missing"),
            "--huggingface-dir",
            str(tmp_path / "hf"),
            "--cover-image",
            str(tmp_path / "cover.png"),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "release directory not found" in captured.err


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_run_packager_does_not_write_on_validation_failure(tmp_path: Path) -> None:
    """Validation failure must NOT leave a corrupt README on disk
    (PR 5.2 self-review #1).

    Forces a validation failure by passing a cover image that is too
    small.  Asserts the README path doesn't materialise and
    ``outcome.errors`` is populated.
    """

    tiny_cover = tmp_path / "tiny.png"
    from PIL import Image

    Image.new("RGB", (10, 10), (0, 0, 0)).save(tiny_cover)
    out_dir = tmp_path / "huggingface"
    outcome = packager.run_packager(_RELEASE_DIR, huggingface_dir=out_dir, cover_image=tiny_cover)
    assert outcome.errors, "expected at least one validation error"
    assert not (out_dir / "README.md").exists(), "README must not be written when validation fails"
    assert outcome.assembled is False


def test_main_rejects_default_config_with_instructor_variant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--default-config`` is meaningless for ``--variant=instructor``
    (only one config); silently accepting it is a misconfiguration
    (PR 5.2 self-review #10).
    """

    rc = packager.main(
        [
            "--release-dir",
            str(_RELEASE_DIR),
            "--huggingface-dir",
            str(tmp_path / "hf"),
            "--variant",
            "instructor",
            "--default-config",
            "advanced",
            "--cover-image",
            str(_COMMITTED_COVER),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "--default-config" in captured.err
    assert "instructor" in captured.err


# ---------------------------------------------------------------------------
# Determinism + sync with committed artefact
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_run_packager_readme_is_byte_deterministic(tmp_path: Path) -> None:
    """Two back-to-back runs against the committed bundles must produce
    byte-identical README files."""

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    packager.run_packager(
        _RELEASE_DIR, huggingface_dir=out_a, cover_image=_COMMITTED_COVER, dry_run=True
    )
    packager.run_packager(
        _RELEASE_DIR, huggingface_dir=out_b, cover_image=_COMMITTED_COVER, dry_run=True
    )
    assert (out_a / "README.md").read_bytes() == (out_b / "README.md").read_bytes()


@pytest.mark.skipif(
    not (_RELEASE_BUNDLES_PRESENT and _COMMITTED_README.exists()),
    reason="release bundles or committed README missing",
)
def test_committed_hf_readme_matches_fresh_regeneration(tmp_path: Path) -> None:
    """A fresh HF README regeneration must match the committed
    ``release/huggingface/README.md`` byte-for-byte AND have a
    non-degenerate body.

    If this fails, ``release/`` drifted without re-running
    ``scripts/package_hf_release.py``.  Regenerate via that script
    from the repo root and commit the new README alongside the bundle
    change.
    """

    fresh_dir = tmp_path / "huggingface"
    packager.run_packager(
        _RELEASE_DIR,
        huggingface_dir=fresh_dir,
        cover_image=_COMMITTED_COVER,
        dry_run=True,
    )
    fresh_bytes = (fresh_dir / "README.md").read_bytes()
    committed_bytes = _COMMITTED_README.read_bytes()
    assert fresh_bytes == committed_bytes

    # Positive content assertions — guard against the failure mode
    # where a code change accidentally produces empty / minimal
    # content that we then re-commit.
    text = fresh_bytes.decode("utf-8")
    # YAML frontmatter parses.
    _, fm_text, body = text.split("---", 2)
    fm = yaml.safe_load(fm_text)
    assert fm["license"] == "mit"
    assert fm["language"] == ["en"]
    assert fm["task_categories"] == ["tabular-classification"]
    assert sorted(fm["tags"]) == fm["tags"]  # rendered sorted
    assert len(fm["configs"]) == 3
    defaults = [c for c in fm["configs"] if c.get("default")]
    assert len(defaults) == 1
    assert defaults[0]["config_name"] == "intermediate"
    # Body inherited the rewritten release card content.
    assert "What's inside" in body
    assert "Why lead scoring matters" in body
    assert "Known limitations" in body
    assert "intermediate_instructor/" not in body  # tree-diagram rewrite fired
    assert "](../" not in body  # parent-link rewrite fired
    assert packager.GITHUB_BLOB_BASE in body


@pytest.mark.skipif(
    not (_INSTRUCTOR_BUNDLE_PRESENT and _COMMITTED_INSTRUCTOR_README.exists()),
    reason="instructor bundle or committed instructor README missing",
)
def test_committed_instructor_readme_matches_fresh_regeneration(tmp_path: Path) -> None:
    """Audit-artifact-sync for the instructor companion.

    A fresh instructor regeneration must match
    ``release/huggingface-instructor/README.md`` byte-for-byte.  Locks
    down the dedicated ``INSTRUCTOR_BODY`` constant introduced in PR
    5.2 self-review #2.
    """

    fresh_dir = tmp_path / "huggingface-instructor"
    packager.run_packager(
        _RELEASE_DIR,
        huggingface_dir=fresh_dir,
        variant="instructor",
        cover_image=_COMMITTED_COVER,
        dry_run=True,
    )
    fresh_bytes = (fresh_dir / "README.md").read_bytes()
    committed_bytes = _COMMITTED_INSTRUCTOR_README.read_bytes()
    assert fresh_bytes == committed_bytes


# ---------------------------------------------------------------------------
# Instructor companion (G12.4)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _INSTRUCTOR_BUNDLE_PRESENT, reason="instructor bundle not present")
def test_run_packager_instructor_variant_packages_independently(tmp_path: Path) -> None:
    """G12.4 — the instructor variant builds a separate upload tree
    pointing at ``release/intermediate_instructor/``, with one config
    (``intermediate``) and ``default: true`` set."""

    upload_dir = tmp_path / "huggingface-instructor"
    outcome = packager.run_packager(
        _RELEASE_DIR,
        huggingface_dir=upload_dir,
        variant="instructor",
        cover_image=_COMMITTED_COVER,
    )
    assert outcome.errors == ()
    assert len(outcome.card.configs) == 1
    only = outcome.card.configs[0]
    assert only.config_name == "intermediate"
    assert only.default is True
    # The intermediate_instructor source dir is flattened to
    # ``intermediate/`` in the upload tree.
    for df in only.data_files:
        assert (upload_dir / df.path).is_file()
    # Instructor body is the dedicated INSTRUCTOR_BODY constant, not
    # the public README inlined verbatim — locks down PR 5.2 self-
    # review fix #2 (3-tier prose was leaking into the 1-tier card).
    body = outcome.card.body
    assert body is packager.INSTRUCTOR_BODY
    # Public-tier names must NOT appear in the instructor body.
    assert "intro" not in body.lower().split()  # word boundary, not substring
    # The instructor body talks about the instructor companion role.
    assert "Instructor companion" in body
    assert "redaction" in body.lower()
    assert "metadata/world_spec" in body
    # Tree block reflects the instructor (single-tier) layout.
    assert "intermediate/" in body
    # No leakage of public tree elements.
    assert "intro/ intermediate/ advanced/" not in body


# ---------------------------------------------------------------------------
# load_dataset() round-trip (G12.3 + G12.4)
#
# Gated on the optional ``datasets`` library — the leadforge dev
# install does not pull it in (the transitive dependency tail is
# heavy).  When present, this test exercises the actual HF loader
# against the assembled upload tree per config and asserts the row
# counts match the manifest.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
@pytest.mark.parametrize("config_name", ["intro", "intermediate", "advanced"])
def test_load_dataset_round_trip_per_config(tmp_path: Path, config_name: str) -> None:
    """G12.3 — ``load_dataset(local_path, name=<config>)`` succeeds for
    every config and returns the canonical row counts."""

    datasets = pytest.importorskip("datasets", reason="datasets SDK not installed")

    upload_dir = tmp_path / "huggingface"
    packager.run_packager(_RELEASE_DIR, huggingface_dir=upload_dir, cover_image=_COMMITTED_COVER)

    ds = datasets.load_dataset(str(upload_dir), name=config_name)
    for hf_split, expected_rows in _EXPECTED_ROWS.items():
        assert hf_split in ds, f"split {hf_split!r} missing from {config_name!r}"
        assert len(ds[hf_split]) == expected_rows, (
            f"{config_name}/{hf_split}: expected {expected_rows} rows, got {len(ds[hf_split])}"
        )


@pytest.mark.skipif(not _INSTRUCTOR_BUNDLE_PRESENT, reason="instructor bundle not present")
def test_load_dataset_round_trip_instructor(tmp_path: Path) -> None:
    """G12.4 — instructor companion loads via ``load_dataset()`` for
    its single config (``intermediate``)."""

    datasets = pytest.importorskip("datasets", reason="datasets SDK not installed")

    upload_dir = tmp_path / "huggingface-instructor"
    packager.run_packager(
        _RELEASE_DIR,
        huggingface_dir=upload_dir,
        variant="instructor",
        cover_image=_COMMITTED_COVER,
    )

    ds = datasets.load_dataset(str(upload_dir), name="intermediate")
    for hf_split, expected_rows in _EXPECTED_ROWS.items():
        assert hf_split in ds
        assert len(ds[hf_split]) == expected_rows
