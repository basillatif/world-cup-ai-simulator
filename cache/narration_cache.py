"""Persistent JSON cache for Claude narration text."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Callable, Literal


VERSION = 1
MODE_GROUP_ANALYSIS = "group_analysis"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_CACHE_PATH = Path(__file__).with_name("group_analysis_cache.json")
CLEAR_CACHE_ENV_VAR = "CLEAR_GROUP_ANALYSIS_CACHE"
LIVE_NARRATION_UNAVAILABLE = "Live narration unavailable (no API key)."

CacheStatus = Literal["hit", "miss", "unavailable"]


def canonical_group_label(group: str) -> str:
    return group if group.startswith("Group ") else f"Group {group}"


def round_probability(value: float) -> float:
    rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(rounded)


def canonical_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    return {
        team: round_probability(probabilities[team])
        for team in sorted(probabilities)
    }


def canonical_payload(
    *,
    group: str,
    probabilities: dict[str, float],
    model: str = DEFAULT_MODEL,
    mode: str = MODE_GROUP_ANALYSIS,
) -> dict:
    return {
        "group": canonical_group_label(group),
        "mode": mode,
        "model": model,
        "probabilities": canonical_probabilities(probabilities),
    }


def cache_key(
    *,
    group: str,
    probabilities: dict[str, float],
    model: str = DEFAULT_MODEL,
    mode: str = MODE_GROUP_ANALYSIS,
) -> str:
    payload = canonical_payload(
        group=group,
        probabilities=probabilities,
        model=model,
        mode=mode,
    )
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def empty_cache(model: str = DEFAULT_MODEL) -> dict:
    return {"version": VERSION, "model": model, "entries": {}}


def load_cache(
    path: Path = DEFAULT_CACHE_PATH,
    *,
    model: str = DEFAULT_MODEL,
) -> dict:
    if os.environ.get(CLEAR_CACHE_ENV_VAR) == "1":
        save_cache(empty_cache(model), path)
        return empty_cache(model)

    if not path.exists() or path.stat().st_size == 0:
        return empty_cache(model)

    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        warnings.warn(
            f"Ignoring corrupt narration cache at {path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return empty_cache(model)

    if not isinstance(cache, dict) or not isinstance(cache.get("entries"), dict):
        warnings.warn(
            f"Ignoring invalid narration cache schema at {path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return empty_cache(model)

    if cache.get("version") != VERSION:
        return empty_cache(model)

    if cache.get("model") != model:
        return empty_cache(model)

    return cache


def save_cache(cache: dict, path: Path = DEFAULT_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def get_entry(
    *,
    group: str,
    probabilities: dict[str, float],
    model: str = DEFAULT_MODEL,
    path: Path = DEFAULT_CACHE_PATH,
) -> dict | None:
    cache = load_cache(path, model=model)
    key = cache_key(group=group, probabilities=probabilities, model=model)
    entry = cache["entries"].get(key)
    if not entry or entry.get("model", model) != model:
        return None
    return entry


def put_entry(
    *,
    group: str,
    probabilities: dict[str, float],
    narration: str,
    model: str = DEFAULT_MODEL,
    path: Path = DEFAULT_CACHE_PATH,
) -> str:
    cache = load_cache(path, model=model)
    key = cache_key(group=group, probabilities=probabilities, model=model)
    snapshot = canonical_probabilities(probabilities)
    cache["entries"][key] = {
        "group": canonical_group_label(group),
        "mode": MODE_GROUP_ANALYSIS,
        "model": model,
        "probabilities": snapshot,
        "narration": narration,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    save_cache(cache, path)
    return key


def get_or_create_group_analysis(
    *,
    group: str,
    probabilities: dict[str, float],
    has_api_key: bool,
    live_narration: Callable[[], str],
    model: str = DEFAULT_MODEL,
    path: Path = DEFAULT_CACHE_PATH,
) -> tuple[str | None, CacheStatus]:
    entry = get_entry(
        group=group,
        probabilities=probabilities,
        model=model,
        path=path,
    )
    if entry:
        return entry["narration"], "hit"

    if not has_api_key:
        return None, "unavailable"

    narration = live_narration()
    put_entry(
        group=group,
        probabilities=probabilities,
        narration=narration,
        model=model,
        path=path,
    )
    return narration, "miss"
