"""The PressureVision checkout has to be nameable without editing the source.

The migration replaced a hardcoded /home/... default with None and left nothing
in its place, so every program that loads the network died at startup on
`None / "config/paper.yml"` -- a TypeError, several frames deep, with no hint
that a path was missing. 787 passing tests did not notice, because none of them
resolved the checkout. These do.
"""

import pytest

import pressurevision_probe as probe


def test_explicit_repo_wins_over_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(probe.PV_REPO_ENV, str(tmp_path / "from-env"))
    assert probe.resolve_repo(tmp_path / "explicit") == tmp_path / "explicit"


def test_environment_names_the_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv(probe.PV_REPO_ENV, str(tmp_path / "clone"))
    assert probe.resolve_repo(None) == tmp_path / "clone"


def test_missing_checkout_says_what_to_set(monkeypatch):
    monkeypatch.delenv(probe.PV_REPO_ENV, raising=False)
    with pytest.raises(SystemExit) as excinfo:
        probe.resolve_repo(None)
    message = str(excinfo.value)
    assert probe.PV_REPO_ENV in message
    assert "--repo" in message


def test_a_directory_that_is_not_a_checkout_is_rejected_before_import(tmp_path, monkeypatch):
    """load_model must not reach `sys.path.insert` or torch with a bogus path."""
    monkeypatch.setenv(probe.PV_REPO_ENV, str(tmp_path))
    with pytest.raises(SystemExit) as excinfo:
        probe.load_model(None, "cpu")
    assert "config/paper.yml" in str(excinfo.value)
