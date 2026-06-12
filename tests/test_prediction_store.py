"""Tests for local prediction persistence."""

from __future__ import annotations

import json

import pandas as pd

from src.simulation.prediction_store import save_predictions


def test_save_predictions_writes_json_and_csv(tmp_path):
    results = {
        "probabilities": {
            "Brazil": {"group_advance": 0.9, "champion": 0.2},
            "France": {"group_advance": 0.85, "champion": 0.25},
        },
        "n_simulations": 5_000,
        "top_contenders": [
            ("France", {"group_advance": 0.85, "champion": 0.25}),
            ("Brazil", {"group_advance": 0.9, "champion": 0.2}),
        ],
    }

    paths = save_predictions(results, seed=42, output_dir=tmp_path)

    assert paths["json"].is_file()
    assert paths["csv"].is_file()
    assert "5000-sims-seed-42" in paths["json"].name

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert payload["n_simulations"] == 5_000
    assert payload["probabilities"] == results["probabilities"]

    probabilities = pd.read_csv(paths["csv"])
    assert probabilities["team"].tolist() == ["France", "Brazil"]
    assert probabilities["champion"].tolist() == [0.25, 0.2]
