from typer.testing import CliRunner

from cli.main import app


runner = CliRunner()


def test_app_help_text_set():
    """The Typer app carries the linear runtime description."""
    assert "linear" in (app.info.help or "").lower()


def test_analyze_help_exposes_linear_options():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--rounds" in result.stdout or "-r" in result.stdout
    assert "--save-report" in result.stdout or "--no-save-repo" in result.stdout


def test_analyze_help_omits_checkpoint_flags():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--checkpoint" not in result.stdout
    assert "--clear-checkpoints" not in result.stdout
