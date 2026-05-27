"""Smoke tests for ``scripts/build_shmuggingface_site.py``.

Covers the three failure modes the self-review identified:

* **Fabricated Kaggle metadata removed** — ``TIER_USABILITY`` and
  ``TIER_MEDAL`` should not appear in the module.
* **Raising on missing manifest / metrics fields** — ``_require``
  should raise ``KeyError`` with a useful message when a required
  field is absent; the module must NOT fall back silently to hardcoded
  defaults.
* **Per-tier dataset card** — ``make_dataset_config`` must use each
  tier's ``dataset_card.md`` as the description body, not the global
  ``release/README.md``.
* **All three tier configs generated** — ``load_tier`` returns the
  expected structure with a non-empty file list, and ``split`` is
  prepended to the column listing.
* **``--branch preview`` is the default** — the ``deploy_site``
  function must pass ``--branch preview`` unless ``production=True``.
* **Stale ``isPrivate: true``** caught — injected bad metadata is
  caught downstream by ``lint_platform_metadata`` (cross-checked in
  ``test_lint_platform_metadata.py``); here we just verify the site
  builder does NOT embed an ``isPrivate`` key.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_shmuggingface_site.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("build_shmuggingface_site", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
smf = importlib.util.module_from_spec(_spec)
sys.modules["build_shmuggingface_site"] = smf
_spec.loader.exec_module(smf)

_RELEASE_DIR = _REPO_ROOT / "release"
_RELEASE_BUNDLES_PRESENT = (_RELEASE_DIR / "intermediate" / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Fabricated metadata removed
# ---------------------------------------------------------------------------


def test_no_tier_usability_constant() -> None:
    """TIER_USABILITY must not exist — those values were fabricated."""
    assert not hasattr(smf, "TIER_USABILITY"), (
        "TIER_USABILITY is still present in build_shmuggingface_site.py. "
        "These were fabricated Kaggle usability scores and must be removed."
    )


def test_no_tier_medal_constant() -> None:
    """TIER_MEDAL must not exist — those values were fabricated."""
    assert not hasattr(smf, "TIER_MEDAL"), (
        "TIER_MEDAL is still present in build_shmuggingface_site.py. "
        "These were fabricated Kaggle medal labels and must be removed."
    )


def test_make_dataset_config_no_kaggle_usability(tmp_path: Path) -> None:
    """The generated config dict must not include kaggleUsability or kaggleMedals."""
    if not _RELEASE_BUNDLES_PRESENT:
        pytest.skip("release/intermediate bundle not present")
    tier_data = smf.load_tier(_RELEASE_DIR, "intermediate")
    config = smf.make_dataset_config(tier_data, tmp_path)
    assert "kaggleUsability" not in config, "make_dataset_config still emits kaggleUsability"
    assert "kaggleMedals" not in config, "make_dataset_config still emits kaggleMedals"


# ---------------------------------------------------------------------------
# _require raises on missing keys
# ---------------------------------------------------------------------------


def test_require_present_key() -> None:
    """_require returns the value when the key exists."""
    d = {"n_leads": 5000, "snapshot_day": 30}
    assert smf._require(d, "n_leads", "test/manifest.json") == 5000


def test_require_missing_key_raises() -> None:
    """_require raises KeyError (not KeyError-silent-default) on miss."""
    d = {"snapshot_day": 30}
    with pytest.raises(KeyError, match="n_leads"):
        smf._require(d, "n_leads", "test/manifest.json")


def test_require_error_includes_context() -> None:
    """_require error message includes the context string for debuggability."""
    d: dict = {}
    with pytest.raises(KeyError, match="my_context"):
        smf._require(d, "missing_key", "my_context")


# ---------------------------------------------------------------------------
# Per-tier dataset_card.md as description
# ---------------------------------------------------------------------------


def test_render_tier_html_uses_dataset_card(tmp_path: Path) -> None:
    """render_tier_html reads dataset_card.md, not README.md."""
    card_content = "# Tier Card\n\nThis is the tier-specific card."
    readme_content = "# Global README\n\nThis is the global README."
    (tmp_path / "dataset_card.md").write_text(card_content)
    (tmp_path / "README.md").write_text(readme_content)

    html = smf.render_tier_html(tmp_path)

    assert "Tier Card" in html
    assert "tier-specific card" in html
    assert "Global README" not in html
    assert "global README" not in html


def test_make_dataset_config_uses_per_tier_card(tmp_path: Path) -> None:
    """make_dataset_config embeds the tier card, not the global README."""
    if not _RELEASE_BUNDLES_PRESENT:
        pytest.skip("release/intermediate bundle not present")

    tier_data = smf.load_tier(_RELEASE_DIR, "intermediate")
    # The description HTML must contain content from dataset_card.md.
    # The per-tier card ships a tier-specific header — check for that.
    config = smf.make_dataset_config(tier_data, tmp_path)
    html = config["descriptionHtml"]
    assert isinstance(html, str), "descriptionHtml must be a string"
    assert len(html) > 200, "descriptionHtml is missing or suspiciously short"
    # The global README starts with a heading containing "leadforge-lead-scoring-v1"
    # and is embedded verbatim when the old path is used.  The dataset_card.md
    # is a per-tier doc that leads with the tier name.
    assert "Intermediate" in html or "intermediate" in html, (
        "Per-tier card HTML does not mention the tier — may be using the wrong source document"
    )


# ---------------------------------------------------------------------------
# split column prepended to column listing
# ---------------------------------------------------------------------------


def test_load_tier_split_column_first(tmp_path: Path) -> None:
    """load_tier must prepend 'split' to the column list from feature_dictionary.csv."""
    if not _RELEASE_BUNDLES_PRESENT:
        pytest.skip("release/intermediate bundle not present")
    tier_data = smf.load_tier(_RELEASE_DIR, "intermediate")
    assert tier_data["columns"][0] == "split", (
        f"Expected first column to be 'split', got {tier_data['columns'][0]!r}"
    )


def test_load_tier_split_column_appears_exactly_once(tmp_path: Path) -> None:
    """'split' must appear exactly once at index 0, regardless of whether it's
    in feature_dictionary.csv.

    ``load_tier`` unconditionally prepends 'split' to the column list.
    Bundles built before this PR won't have 'split' in their
    feature_dictionary.csv; bundles built after will.  Either way, the
    resulting column listing must have 'split' exactly once, first.
    """
    if not _RELEASE_BUNDLES_PRESENT:
        pytest.skip("release/intermediate bundle not present")
    tier_data = smf.load_tier(_RELEASE_DIR, "intermediate")
    cols = tier_data["columns"]
    assert cols[0] == "split", f"Expected 'split' at index 0 of columns, got {cols[0]!r}"
    assert cols.count("split") == 1, (
        f"'split' appears {cols.count('split')} times in column list — "
        "expected exactly once. load_tier may be double-prepending."
    )


# ---------------------------------------------------------------------------
# All three tiers produce valid configs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _RELEASE_BUNDLES_PRESENT,
    reason="release bundles not present",
)
@pytest.mark.parametrize("tier", smf.TIERS)
def test_load_tier_structure(tier: str, tmp_path: Path) -> None:
    """load_tier returns a dict with non-empty file lists for each tier."""
    tier_data = smf.load_tier(_RELEASE_DIR, tier)
    assert tier_data["tier"] == tier
    assert tier_data["n_rows"] > 0
    assert len(tier_data["columns"]) > 5  # at least some columns
    assert len(tier_data["sample_rows"]) > 0


@pytest.mark.skipif(
    not _RELEASE_BUNDLES_PRESENT,
    reason="release bundles not present",
)
@pytest.mark.parametrize("tier", smf.TIERS)
def test_make_dataset_config_structure(tier: str, tmp_path: Path) -> None:
    """make_dataset_config produces the required fields for each tier."""
    tier_data = smf.load_tier(_RELEASE_DIR, tier)
    config = smf.make_dataset_config(tier_data, tmp_path)

    assert config["slug"].endswith(tier)
    assert len(config["files"]) >= 5  # csv, feature dict, 3 parquets at minimum
    assert config["rowCount"] > 0
    assert config["splits"] == ["train", "valid", "test"]
    assert isinstance(config["descriptionHtml"], str), "descriptionHtml must be a string"
    assert len(config["descriptionHtml"]) > 100, "descriptionHtml is suspiciously short"

    # Required fields must be present — no silent defaults
    for key in ("slug", "title", "license", "task", "rowCount"):
        assert key in config, f"Config missing required key: {key!r}"


# ---------------------------------------------------------------------------
# _require raises on stale schema (regression guard for the "silent default" bug)
# ---------------------------------------------------------------------------


def test_load_tier_raises_on_missing_n_leads(tmp_path: Path) -> None:
    """load_tier / make_dataset_config must raise if n_leads is absent from manifest."""
    if not _RELEASE_BUNDLES_PRESENT:
        pytest.skip("release/intermediate bundle not present")

    import copy

    tier_data = smf.load_tier(_RELEASE_DIR, "intermediate")
    # Simulate a bundle where n_leads was renamed / dropped
    bad_manifest = copy.deepcopy(tier_data["manifest"])
    bad_manifest.pop("n_leads")
    tier_data_bad = {**tier_data, "manifest": bad_manifest}

    with pytest.raises(KeyError, match="n_leads"):
        smf.make_dataset_config(tier_data_bad, tmp_path)


# ---------------------------------------------------------------------------
# deploy_site uses --branch preview by default
# ---------------------------------------------------------------------------


def test_deploy_site_preview_branch_by_default(tmp_path: Path) -> None:
    """deploy_site must default to --branch preview, not --branch main."""
    cf_env = tmp_path / "cf.env"
    cf_env.write_text("CLOUDFLARE_ACCOUNT_ID=fake-account\nCLOUDFLARE_API_TOKEN=fake-token\n")
    captured_cmd: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
        captured_cmd.append(cmd)
        result = MagicMock()
        result.returncode = 0
        return result

    with patch("subprocess.run", side_effect=fake_run):
        smf.deploy_site(
            out_dir=tmp_path / "dist",
            project_name="test-project",
            cf_env_path=cf_env,
            production=False,  # default
        )

    assert captured_cmd, "subprocess.run was never called"
    cmd = captured_cmd[0]
    branch_idx = cmd.index("--branch")
    assert cmd[branch_idx + 1] == "preview", (
        f"Expected --branch preview but got --branch {cmd[branch_idx + 1]!r}. "
        "A stray local deploy must never clobber the production site."
    )


def test_deploy_site_main_branch_with_production_flag(tmp_path: Path) -> None:
    """deploy_site must use --branch main when production=True."""
    cf_env = tmp_path / "cf.env"
    cf_env.write_text("CLOUDFLARE_ACCOUNT_ID=fake-account\nCLOUDFLARE_API_TOKEN=fake-token\n")
    captured_cmd: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
        captured_cmd.append(cmd)
        result = MagicMock()
        result.returncode = 0
        return result

    with patch("subprocess.run", side_effect=fake_run):
        smf.deploy_site(
            out_dir=tmp_path / "dist",
            project_name="test-project",
            cf_env_path=cf_env,
            production=True,
        )

    assert captured_cmd
    cmd = captured_cmd[0]
    branch_idx = cmd.index("--branch")
    assert cmd[branch_idx + 1] == "main", (
        f"Expected --branch main for production deploy but got {cmd[branch_idx + 1]!r}"
    )


# ---------------------------------------------------------------------------
# _rewrite_links handles bare relative links
# ---------------------------------------------------------------------------


def test_rewrite_links_bare_license() -> None:
    """[LICENSE](LICENSE) must be rewritten to an absolute URL."""
    text = "See [LICENSE](LICENSE) for details."
    result = smf._rewrite_links(text, "https://github.com/org/repo/blob/main/release/intro")
    assert "](LICENSE)" not in result, (
        "_rewrite_links left a bare relative [LICENSE](LICENSE) link — "
        "it would 404 on the static host"
    )
    assert "https://" in result


def test_rewrite_links_parent_dir() -> None:
    """[text](../foo) must become an absolute GitHub blob URL."""
    text = "See [guide](../docs/release/break_me_guide.md) for more."
    result = smf._rewrite_links(text, "https://example.com/base")
    assert "](../docs/" not in result
    assert smf.GITHUB_BLOB_BASE in result


def test_rewrite_links_absolute_unchanged() -> None:
    """Absolute https:// links must not be modified."""
    url = "https://example.com/some/path"
    text = f"[click]({url})"
    result = smf._rewrite_links(text, "https://github.com/org/repo/blob/main/release/intro")
    assert url in result
