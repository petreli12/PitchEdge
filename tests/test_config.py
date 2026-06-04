"""Config loads with sensible, correctly-typed defaults (no DB required)."""

from __future__ import annotations

import importlib


def test_defaults_are_typed(monkeypatch):
    for var in ("RANDOM_SEED", "BLEND_W", "MODEL_TEMPERATURE", "DB_URL"):
        monkeypatch.delenv(var, raising=False)
    import pitchedge.config as config

    config = importlib.reload(config)

    assert isinstance(config.RANDOM_SEED, int)
    assert isinstance(config.BLEND_W, float)
    assert 0.0 <= config.BLEND_W <= 1.0
    assert config.MODEL_TEMPERATURE > 0.0
    assert config.DB_URL.startswith("postgresql")


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("BLEND_W", "0.75")
    monkeypatch.setenv("RANDOM_SEED", "42")
    import pitchedge.config as config

    config = importlib.reload(config)

    assert config.BLEND_W == 0.75
    assert config.RANDOM_SEED == 42
