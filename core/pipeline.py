# -*- coding: utf-8 -*-
"""Пайплайн загрузки, отбора пары и бэктеста (без зависимости от Streamlit)."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional

import pandas as pd

from core.cointegration import CointegrationTester
from core.correlation import CorrelationAnalyzer
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


def _build_backtest_details(bt_result: Dict, trades: pd.DataFrame) -> Dict:
    """Расширенная интерпретация бэктеста для UI."""
    returns = bt_result["returns"].dropna()
    if returns.empty:
        return {
            "volatility_daily": 0.0,
            "best_day": 0.0,
            "worst_day": 0.0,
            "avg_holding_days": 0.0,
            "median_holding_days": 0.0,
            "trade_win_rate": 0.0,
            "avg_trade_pnl": 0.0,
        }

    if trades.empty:
        return {
            "volatility_daily": float(returns.std()),
            "best_day": float(returns.max()),
            "worst_day": float(returns.min()),
            "avg_holding_days": 0.0,
            "median_holding_days": 0.0,
            "trade_win_rate": 0.0,
            "avg_trade_pnl": 0.0,
        }

    return {
        "volatility_daily": float(returns.std()),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "avg_holding_days": float(trades["holding_days"].mean()),
        "median_holding_days": float(trades["holding_days"].median()),
        "trade_win_rate": float((trades["pnl"] > 0).mean()),
        "avg_trade_pnl": float(trades["pnl"].mean()),
    }


def _get_top_correlation_pair(prices: pd.DataFrame) -> Optional[Dict]:
    """Находит пару с максимальной |корреляцией| и строит хедж-коэффициент по лог-ценам."""
    analyzer = CorrelationAnalyzer(prices)
    corr_matrix = analyzer.compute_correlation_matrix()
    tickers = prices.columns.tolist()

    best_pair = None
    best_corr = None
    for t1, t2 in combinations(tickers, 2):
        corr = corr_matrix.loc[t1, t2]
        if best_corr is None or abs(corr) > abs(best_corr):
            best_corr = corr
            best_pair = (t1, t2)

    if best_pair is None:
        return None

    pair_df = prices[list(best_pair)].dropna()
    if len(pair_df) < 30:
        return None

    x = pair_df.iloc[:, 0]
    y = pair_df.iloc[:, 1]
    beta = float((y / x).median()) if (x != 0).all() else 1.0
    spread = y - beta * x

    return {
        "pair": best_pair,
        "beta": beta,
        "spread": spread,
        "correlation": float(best_corr),
    }


def load_and_prepare_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    missing_threshold: float,
    use_cache: bool,

) -> tuple[pd.DataFrame, Dict]:
    """Загружает данные MOEX и выполняет базовую очистку/синхронизацию."""

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

    # 1) Статистический кандидат (как было раньше).
    best_pair = tester.get_best_pair()
    if best_pair is None:
        return None

    # 2) Торговый кандидат: выбираем из tradable-пар ту,
    #    у которой лучший Sharpe в "быстром" бэктесте на тех же параметрах.
    tradable_pairs = [p for p in tester.results if tester._is_tradable_pair(p)]  # noqa: SLF001
    best_pair_by_bt = None
    best_pair_score = None

    for candidate in tradable_pairs:
        candidate_strategy = PairsTradingStrategy(
            spread=candidate["spread"],
            window=z_window,
            entry_z=entry_z,
            exit_z=exit_z,
        )
        candidate_signals = candidate_strategy.generate_signals(max_holding_days=max_holding_days)
        candidate_returns = _build_pair_returns(prices=prices, best_pair=candidate)

        candidate_bt = Backtest(
            signals=candidate_signals,
            spread=candidate["spread"],
            pair_returns=candidate_returns,
            initial_capital=1.0,
        )
        candidate_result = candidate_bt.run()
        m = candidate_result["metrics"]

        # Отбрасываем пары без сделок/сигналов.
        if m["num_trades"] <= 0:
            continue

        score = (m["sharpe_ratio"], m["total_return"])
        if best_pair_score is None or score > best_pair_score:
            best_pair_score = score
            best_pair_by_bt = candidate

    if best_pair_by_bt is not None:
        best_pair = best_pair_by_bt

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
    coint_details = _build_backtest_details(bt_result=bt_result, trades=trades)

    corr_pair = _get_top_correlation_pair(prices=prices)
    correlation_backtest = None
    best_method = "cointegration"
    comparison_reason = "Корреляционный бенчмарк недоступен для выбранной выборки."

    if corr_pair is not None:
        corr_strategy = PairsTradingStrategy(
            spread=corr_pair["spread"],
            window=z_window,
            entry_z=entry_z,
            exit_z=exit_z,
        )
        corr_signals = corr_strategy.generate_signals(max_holding_days=max_holding_days)
        corr_trades = corr_strategy.get_trades()
        corr_returns = _build_pair_returns(
            prices=prices,
            best_pair={"pair": corr_pair["pair"], "beta": corr_pair["beta"]},
        )
        corr_backtest = Backtest(
            signals=corr_signals,
            spread=corr_pair["spread"],
            pair_returns=corr_returns,
            initial_capital=1.0,
        )
        corr_result = corr_backtest.run()
        corr_details = _build_backtest_details(bt_result=corr_result, trades=corr_trades)
        correlation_backtest = {
            "pair": corr_pair,
            "signals": corr_signals,
            "trades": corr_trades,
            "metrics": corr_result["metrics"],
            "equity": corr_result["cumulative_returns"],
            "details": corr_details,
        }

        coint_metrics = bt_result["metrics"]
        corr_metrics = corr_result["metrics"]
        coint_score = (coint_metrics["sharpe_ratio"], coint_metrics["total_return"], coint_metrics["max_drawdown"])
        corr_score = (corr_metrics["sharpe_ratio"], corr_metrics["total_return"], corr_metrics["max_drawdown"])
        best_method = "cointegration" if coint_score >= corr_score else "correlation"
        comparison_reason = (
            "Сравнение по приоритету: Sharpe → Total Return → Max Drawdown. "
            f"Победитель: {best_method}."
        )

    return {
        "best_pair": best_pair,
        "signals": signals,
        "trades": trades,
        "metrics": bt_result["metrics"],
        "equity": bt_result["cumulative_returns"],
        "details": coint_details,
        "correlation_backtest": correlation_backtest,
        "comparison_table": tester.get_comparison_table(),
        "best_method": best_method,
        "comparison_reason": comparison_reason,
    }
