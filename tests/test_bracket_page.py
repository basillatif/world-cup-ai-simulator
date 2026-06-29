from pathlib import Path

from knockout_bracket import simulate_bracket
from knockout_engine import knockout_match_fn
from src.app.streamlit_app import prepare_bracket_data


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "src" / "app" / "streamlit_app.py"


def test_bracket_data_prep_imports_and_runs_without_streamlit_runtime():
    table = prepare_bracket_data(n=5)

    assert len(table) == 32
    assert ["Team", "Champion %", "Reach SF %"] == list(table.columns[:3])


def test_knockout_simulation_returns_32_teams_and_conserves_champion_probability():
    results = simulate_bracket(knockout_match_fn, n=200)

    assert len(results) == 32
    total = sum(team_result["champion"] for team_result in results.values())
    assert abs(total - 1.0) <= 0.05


def test_entrypoint_has_no_anthropic_import_or_reference():
    assert "anthropic" not in ENTRYPOINT.read_text().casefold()
