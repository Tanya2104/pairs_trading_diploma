# -*- coding: utf-8 -*-
"""Пайплайн загрузки, отбора пары и бэктеста (без зависимости от Streamlit)."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from core.cointegration import CointegrationTester
from core.data_loader import MOEXLoader
from core.data_processor import DataProcessor
from strategy.backtest import Backtest
from strategy.signals import PairsTradingStrategy


def _build_pair_returns(prices: pd.DataFrame, best_pair: Dict) -> pd.Series:
    """Собирает доходность рыночно-нейтрального портфеля пары: r_y - beta * r_x."""
    ticker_x, ticker_y = best_pair["pair"]
    beta = best_pair["beta"]

    if ticker_x not in prices.columns or ticker_y not in prices.columns:
        raise KeyError(f"Tickers {ticker_x}/{ticker_y} are missing in prices DataFrame.")

    pair_returns = prices[ticker_y].pct_change() - beta * prices[ticker_x].pct_change()
    return pair_returns.reindex(prices.index).fillna(0.0)


def load_and_prepare_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    missing_threshold: float,
    use_cache: bool,

) -> tuple[pd.DataFrame, Dict]:
    """Загружает данные MOEX и выполняет базовую очистку/синхронизацию."""
)   tuple[pd.DataFrame, Dict]:

    loader = MOEXLoader(use_cache=use_cache)
    raw_prices = loader.load_prices(tickers=tickers, start_date=start_date, end_date=end_date)

    processor = DataProcessor(raw_prices)
    quality = processor.check_quality()
    cleaned = processor.remove_empty_tickers(threshold=missing_threshold)
    processed = processor.synchronize_dates() if cleaned is not None else None

    if processed is None or processed.empty:
        raise ValueError("После обработки не осталось данных. Проверьте тикеры/порог пропусков.")

    return processed, quality


def run_full_pipeline(
    prices: pd.DataFrame,
    p_value_threshold: float,
    z_window: int,
    entry_z: float,
    exit_z: float,
    max_holding_days: int,
) -> Optional[Dict]:
    """Запускает поиск пары, генерацию сигналов и бэктест; возвращает словарь результатов."""
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

    pair_returns = _build_pair_returns(prices=prices, best_pair=best_pair)

    backtest = Backtest(
        signals=signals,
        spread=best_pair["spread"],
        pair_returns=pair_returns,
        initial_capital=1.0,
    )
    bt_result = backtest.run()

    return {
        "best_pair": best_pair,
        "signals": signals,
        "trades": trades,
        "metrics": bt_result["metrics"],
        "equity": bt_result["cumulative_returns"],
    }
