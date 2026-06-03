from __future__ import annotations

import pytest
from typer.testing import CliRunner

from reflex.cli import app
from reflex.version import __version__

runner = CliRunner()


@pytest.fixture
def cfg_file(tmp_path):
    p = tmp_path / "cfg.yaml"
    db = (tmp_path / "cli.db").as_posix()
    p.write_text(
        f"memory:\n  db_path: {db}\n  vector:\n    backend: numpy\n"
        "embeddings:\n  dim: 128\n"
        "logging:\n  rich: false\n  level: ERROR\n"
    )
    return p


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_config_show(cfg_file) -> None:
    result = runner.invoke(app, ["config", "-c", str(cfg_file)])
    assert result.exit_code == 0
    assert "memory:" in result.stdout


def test_chat(cfg_file) -> None:
    result = runner.invoke(
        app, ["chat", "Remember that reflex is a memory system.", "-c", str(cfg_file)]
    )
    assert result.exit_code == 0
    assert "integrity=" in result.stdout


def test_inspect(cfg_file) -> None:
    # First write something, then inspect the same DB.
    runner.invoke(app, ["chat", "Remember the build server is fast.", "-c", str(cfg_file)])
    result = runner.invoke(app, ["inspect", "-c", str(cfg_file)])
    assert result.exit_code == 0
    assert "episodic" in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    # `no_args_is_help=True` prints help and exits with Click's usage code (2).
    assert result.exit_code in (0, 2)
    assert "Usage" in result.output or "Commands" in result.output
