"""Local persistence for Monte Carlo tournament predictions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PREDICTIONS_DIR = Path(__file__).parents[2] / "pre-wc-predictions"


def save_predictions(
    results: dict[str, Any],
    *,
    seed: int,
    output_dir: Path = DEFAULT_PREDICTIONS_DIR,
) -> dict[str, Path]:
    """Save full simulation results as JSON and team probabilities as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc)
    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    n_simulations = int(results["n_simulations"])
    stem = f"predictions-{timestamp}-{n_simulations}-sims-seed-{seed}"

    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    payload = {
        "generated_at": generated_at.isoformat(),
        "seed": seed,
        **results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    probabilities = pd.DataFrame.from_dict(
        results["probabilities"], orient="index"
    )
    probabilities.index.name = "team"
    probabilities.sort_values("champion", ascending=False).to_csv(csv_path)

    return {"json": json_path, "csv": csv_path}
