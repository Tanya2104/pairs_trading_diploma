# -*- coding: utf-8 -*-
"""Smoke-тест пайплайна (совместим с запуском `python core/smoke_pipeline.py`)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.pipeline import run_full_pipeline


def build_synthetic_prices(n_days: int = 180, seed: int = 42) -> pd.DataFrame:
    """Создаёт синтетические коинтегрированные ряды цен для smoke-теста."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_days, freq="B")
    base = np.cumsum(rng.normal(0, 1, size=n_days)) + 100
    s1 = base + rng.normal(0, 0.3, size=n_days)
    s2 = base * 1.05 + rng.normal(0, 0.3, size=n_days)
    return pd.DataFrame({"AAA": s1, "BBB": s2}, index=index)


def main() -> None:
    """Запускает smoke-тест и валидирует структуру результата."""
    prices = build_synthetic_prices()
    result = run_full_pipeline(
        prices=prices,
        p_value_threshold=0.1,
        z_window=20,
        entry_z=1.5,
        exit_z=0.2,
        max_holding_days=20,
    )

    if result is None:
        raise RuntimeError(
            "Smoke-тест не прошёл: пара не найдена на синтетических данных"
        )

    required_keys = {"best_pair", "signals", "trades", "metrics", "equity"}
    if not required_keys.issubset(result.keys()):
        raise RuntimeError(
            f"Smoke-тест не прошёл: отсутствуют ключи {required_keys - set(result.keys())}"
        )

    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
