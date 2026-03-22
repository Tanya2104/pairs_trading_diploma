# -*- coding: utf-8 -*-
"""Smoke-РЎвЂљР ВµРЎРѓРЎвЂљ Р С—Р В°Р в„–Р С—Р В»Р В°Р в„–Р Р…Р В° (РЎРѓР С•Р Р†Р СР ВµРЎРѓРЎвЂљР С‘Р С РЎРѓ Р В·Р В°Р С—РЎС“РЎРѓР С”Р С•Р С `python core/smoke_pipeline.py`)."""

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
    """Р РЋР С•Р В·Р Т‘Р В°РЎвЂРЎвЂљ РЎРѓР С‘Р Р…РЎвЂљР ВµРЎвЂљР С‘РЎвЂЎР ВµРЎРѓР С”Р С‘Р Вµ Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р Р…РЎвЂ№Р Вµ РЎР‚РЎРЏР Т‘РЎвЂ№ РЎвЂ Р ВµР Р… Р Т‘Р В»РЎРЏ smoke-РЎвЂљР ВµРЎРѓРЎвЂљР В°."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_days, freq="B")
    base = np.cumsum(rng.normal(0, 1, size=n_days)) + 100
    s1 = base + rng.normal(0, 0.3, size=n_days)
    s2 = base * 1.05 + rng.normal(0, 0.3, size=n_days)
    return pd.DataFrame({"AAA": s1, "BBB": s2}, index=index)


def main() -> None:
    """Р вЂ”Р В°Р С—РЎС“РЎРѓР С”Р В°Р ВµРЎвЂљ smoke-РЎвЂљР ВµРЎРѓРЎвЂљ Р С‘ Р Р†Р В°Р В»Р С‘Р Т‘Р С‘РЎР‚РЎС“Р ВµРЎвЂљ РЎРѓРЎвЂљРЎР‚РЎС“Р С”РЎвЂљРЎС“РЎР‚РЎС“ РЎР‚Р ВµР В·РЎС“Р В»РЎРЉРЎвЂљР В°РЎвЂљР В°."""
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
        raise RuntimeError("Smoke-РЎвЂљР ВµРЎРѓРЎвЂљ Р Р…Р Вµ Р С—РЎР‚Р С•РЎв‚¬РЎвЂР В»: Р С—Р В°РЎР‚Р В° Р Р…Р Вµ Р Р…Р В°Р в„–Р Т‘Р ВµР Р…Р В° Р Р…Р В° РЎРѓР С‘Р Р…РЎвЂљР ВµРЎвЂљР С‘РЎвЂЎР ВµРЎРѓР С”Р С‘РЎвЂ¦ Р Т‘Р В°Р Р…Р Р…РЎвЂ№РЎвЂ¦")

    required_keys = {"best_pair", "signals", "trades", "metrics", "equity"}
    if not required_keys.issubset(result.keys()):
        raise RuntimeError(f"Smoke-РЎвЂљР ВµРЎРѓРЎвЂљ Р Р…Р Вµ Р С—РЎР‚Р С•РЎв‚¬РЎвЂР В»: Р С•РЎвЂљРЎРѓРЎС“РЎвЂљРЎРѓРЎвЂљР Р†РЎС“РЎР‹РЎвЂљ Р С”Р В»РЎР‹РЎвЂЎР С‘ {required_keys - set(result.keys())}")

    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
