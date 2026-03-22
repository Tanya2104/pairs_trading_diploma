# -*- coding: utf-8 -*-
"""РџР°Р№РїР»Р°Р№РЅ Р·Р°РіСЂСѓР·РєРё, РѕС‚Р±РѕСЂР° РїР°СЂС‹ Рё Р±СЌРєС‚РµСЃС‚Р° (Р±РµР· Р·Р°РІРёСЃРёРјРѕСЃС‚Рё РѕС‚ Streamlit)."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from core.cointegration import CointegrationTester
from core.data_loader import MOEXLoader
from core.data_processor import DataProcessor
from strategy.backtest import Backtest
from strategy.signals import PairsTradingStrategy


def load_and_prepare_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    missing_threshold: float,
    use_cache: bool,
) -> tuple[pd.DataFrame, Dict]:
    """Р—Р°РіСЂСѓР¶Р°РµС‚ РґР°РЅРЅС‹Рµ MOEX Рё РІС‹РїРѕР»РЅСЏРµС‚ Р±Р°Р·РѕРІСѓСЋ РѕС‡РёСЃС‚РєСѓ/СЃРёРЅС…СЂРѕРЅРёР·Р°С†РёСЋ."""
    loader = MOEXLoader(use_cache=use_cache)
    raw_prices = loader.load_prices(tickers=tickers, start_date=start_date, end_date=end_date)

    processor = DataProcessor(raw_prices)
    quality = processor.check_quality()
    cleaned = processor.remove_empty_tickers(threshold=missing_threshold)
    processed = processor.synchronize_dates() if cleaned is not None else None

    if processed is None or processed.empty:
        raise ValueError("РџРѕСЃР»Рµ РѕР±СЂР°Р±РѕС‚РєРё РЅРµ РѕСЃС‚Р°Р»РѕСЃСЊ РґР°РЅРЅС‹С…. РџСЂРѕРІРµСЂСЊС‚Рµ С‚РёРєРµСЂС‹/РїРѕСЂРѕРі РїСЂРѕРїСѓСЃРєРѕРІ.")

    return processed, quality


def run_full_pipeline(
    prices: pd.DataFrame,
    p_value_threshold: float,
    z_window: int,
    entry_z: float,
    exit_z: float,
    max_holding_days: int,
) -> Optional[Dict]:
    """Р—Р°РїСѓСЃРєР°РµС‚ РїРѕРёСЃРє РїР°СЂС‹, РіРµРЅРµСЂР°С†РёСЋ СЃРёРіРЅР°Р»РѕРІ Рё Р±СЌРєС‚РµСЃС‚; РІРѕР·РІСЂР°С‰Р°РµС‚ СЃР»РѕРІР°СЂСЊ СЂРµР·СѓР»СЊС‚Р°С‚РѕРІ."""
    tester = CointegrationTester(prices=prices, p_value_threshold=p_value_threshold)
    tester.find_pairs()
    best_pair = tester.get_best_pair()

    if best_pair is None:
        return None

    strategy = PairsTradingStrategy(
        spread=best_pair["spread"],
        window=z_window,
        entry_z=entry_z,
        exit_z=exit_z,
    )
    signals = strategy.generate_signals(max_holding_days=max_holding_days)
    trades = strategy.get_trades()

    backtest = Backtest(signals=signals, spread=best_pair["spread"], initial_capital=1.0)
    bt_result = backtest.run()

    return {
        "best_pair": best_pair,
        "signals": signals,
        "trades": trades,
        "metrics": bt_result["metrics"],
        "equity": bt_result["cumulative_returns"],
    }
