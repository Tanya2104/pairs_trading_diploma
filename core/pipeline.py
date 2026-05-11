# -*- coding: utf-8 -*-
"""Пайплайн загрузки, отбора пары и бэктеста (без зависимости от Streamlit)."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from core.cointegration import CointegrationTester
from core.correlation import CorrelationAnalyzer
from core.data_loader import MOEXLoader
from core.data_processor import DataProcessor
from core.spread_analysis import calculate_spread, calculate_zscore, plot_spread, plot_zscore, save_spread_analysis, spread_statistics
from strategy.backtest import Backtest
from strategy.signals import PairsTradingStrategy


COMPANY_NAMES_MAP = {
    "SBER": "ПАО Сбербанк",
    "GAZP": "ПАО Газпром",
    "LKOH": "ПАО ЛУКОЙЛ",
    "ROSN": "ПАО НК Роснефть",
    "NVTK": "ПАО НОВАТЭК",
    "TATN": "ПАО Татнефть",
    "GMKN": "ПАО ГМК Норильский никель",
    "PLZL": "ПАО Полюс",
    "CHMF": "ПАО Северсталь",
    "NLMK": "ПАО НЛМК",
    "MAGN": "ПАО ММК",
    "YNDX": "МКПАО Яндекс",
    "MTSS": "ПАО МТС",
    "AFLT": "ПАО Аэрофлот",
    "VTBR": "Банк ВТБ (ПАО)",
}


def prepare_experimental_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    use_cache: bool,
    output_dir: str = "data/results",
) -> Dict:
    """Подготовка и описание экспериментальных данных для раздела 3.1 ВКР."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loader = MOEXLoader(use_cache=use_cache)
    prices_raw = loader.load_prices(tickers=tickers, start_date=start_date, end_date=end_date)
    if prices_raw.empty:
        raise ValueError("Не удалось загрузить данные по выбранным тикерам за указанный период.")

    prices_raw = prices_raw.sort_index()
    n_rows_before = len(prices_raw)
    missing_by_ticker = prices_raw.isna().sum().astype(int)

    prices_clean = prices_raw.sort_index().dropna(how="any")
    if prices_clean.empty:
        raise ValueError("После синхронизации и удаления пропусков данные отсутствуют.")

    n_rows_after = len(prices_clean)
    sample_start = prices_clean.index.min()
    sample_end = prices_clean.index.max()

    stats_rows = []
    for ticker in prices_clean.columns:
        series = prices_clean[ticker]
        stats_rows.append(
            {
                "Тикер": ticker,
                "Название компании": COMPANY_NAMES_MAP.get(ticker, ticker),
                "Количество наблюдений": int(series.count()),
                "Минимальная цена": float(series.min()),
                "Максимальная цена": float(series.max()),
                "Средняя цена": float(series.mean()),
                "Стандартное отклонение": float(series.std()),
            }
        )
    stats_df = pd.DataFrame(stats_rows)

    prices_csv_path = output_path / "prepared_closing_prices.csv"
    stats_csv_path = output_path / "descriptive_statistics.csv"
    plot_png_path = output_path / "closing_prices_dynamics.png"

    prices_clean.to_csv(prices_csv_path, index_label="Дата")
    stats_df.to_csv(stats_csv_path, index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in prices_clean.columns:
        ax.plot(prices_clean.index, prices_clean[ticker], label=ticker, linewidth=1.2)
    ax.set_title("Динамика цен закрытия акций за выбранный период")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Цена закрытия")
    ax.legend(loc="best", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_png_path, dpi=150)
    plt.close(fig)

    quality = {
        "Количество строк до очистки": int(n_rows_before),
        "Количество строк после очистки": int(n_rows_after),
        "Пропуски по тикерам": missing_by_ticker.to_dict(),
        "Дата начала выборки": sample_start.strftime("%Y-%m-%d"),
        "Дата окончания выборки": sample_end.strftime("%Y-%m-%d"),
    }

    return {
        "prices_raw": prices_raw,
        "prices_clean": prices_clean,
        "quality": quality,
        "head": prices_clean.head(10),
        "stats": stats_df,
        "files": {
            "prices_csv": str(prices_csv_path),
            "stats_csv": str(stats_csv_path),
            "plot_png": str(plot_png_path),
        },
    }


def _build_pair_returns(prices: pd.DataFrame, best_pair: Dict) -> pd.Series:
    """Собирает доходность рыночно-нейтрального портфеля пары: r_y - beta * r_x."""
    ticker_x, ticker_y = best_pair["pair"]
    beta = best_pair["beta"]

    if ticker_x not in prices.columns or ticker_y not in prices.columns:
        raise KeyError(f"Tickers {ticker_x}/{ticker_y} are missing in prices DataFrame.")

    pair_returns = prices[ticker_y].pct_change() - beta * prices[ticker_x].pct_change()
    return pair_returns.reindex(prices.index).fillna(0.0)


def _build_backtest_details(backtest_result: Dict, trades: pd.DataFrame) -> Dict:
    """Расширенная интерпретация бэктеста для UI."""
    returns = backtest_result["returns"].dropna()
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

    has_pnl = "pnl" in trades.columns
    has_holding_days = "holding_days" in trades.columns

    if trades.empty or not has_pnl:
        return {
            "volatility_daily": float(returns.std()),
            "best_day": float(returns.max()),
            "worst_day": float(returns.min()),
            "avg_holding_days": float(trades["holding_days"].mean()) if has_holding_days and not trades.empty else 0.0,
            "median_holding_days": float(trades["holding_days"].median()) if has_holding_days and not trades.empty else 0.0,
            "trade_win_rate": 0.0,
            "avg_trade_pnl": 0.0,
        }

    return {
        "volatility_daily": float(returns.std()),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "avg_holding_days": float(trades["holding_days"].mean()) if has_holding_days else 0.0,
        "median_holding_days": float(trades["holding_days"].median()) if has_holding_days else 0.0,
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




def _run_strategy_backtest_for_pair(
    prices: pd.DataFrame,
    pair: tuple[str, str],
    beta: float,
    spread: pd.Series,
    z_window: int,
    entry_z: float,
    exit_z: float,
    max_holding_days: int,
) -> Dict:
    """Единый пайплайн стратегия → сигналы → сделки → бэктест для выбранной пары."""
    strategy = PairsTradingStrategy(spread=spread, window=z_window, entry_z=entry_z, exit_z=exit_z)
    signals = strategy.generate_signals(max_holding_days=max_holding_days)
    trades = strategy.get_trades()
    pair_returns = _build_pair_returns(prices=prices, best_pair={"pair": pair, "beta": beta})
    backtest = Backtest(signals=signals, spread=spread, pair_returns=pair_returns, initial_capital=1.0)
    raw_backtest = backtest.run()
    equity_curve = raw_backtest.get("cumulative_returns")
    if equity_curve is None:
        equity_curve = pd.Series(dtype=float)

    peak = equity_curve.expanding().max() if len(equity_curve) else equity_curve
    drawdown = (equity_curve - peak) / peak if len(equity_curve) else pd.Series(dtype=float)

    metrics = dict(raw_backtest.get("metrics", {}))
    metrics["final_capital"] = float(equity_curve.iloc[-1]) if len(equity_curve) else 1.0

    backtest_result = {
        "returns": raw_backtest.get("returns", pd.Series(dtype=float)),
        "equity_curve": equity_curve,
        "cumulative_returns": equity_curve,
        "drawdown": drawdown.fillna(0.0) if len(drawdown) else drawdown,
        "metrics": metrics,
        "trades": trades,
    }

    details = _build_backtest_details(backtest_result=backtest_result, trades=trades)
    return {
        "signals": signals,
        "backtest_result": backtest_result,
        "details": details,
    }


def _save_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: str = "data/results/heatmap_correlation.png") -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Корреляционная матрица цен закрытия")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path

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
    selected_pair: Optional[tuple[str, str]] = None,
    selection_mode: str = "auto",
) -> Optional[Dict]:
    """Запускает поиск пары, генерацию сигналов и бэктест; возвращает словарь результатов."""
    tester = CointegrationTester(prices=prices, p_value_threshold=p_value_threshold)
    tester.find_pairs()
    coint_results_df = tester.results_to_dataframe()
    coint_saved_files = tester.save_results()
    coint_heatmap_path = tester.save_pvalue_heatmap()

    if coint_results_df.empty:
        return None

    best_pair = None
    pair_source = "cointegration:min_p_value"
    if selection_mode == "manual" and selected_pair is not None:
        for candidate in tester.results:
            if tuple(candidate["pair"]) == tuple(selected_pair):
                best_pair = candidate
                pair_source = "cointegration:manual_user_selection"
                break
    if best_pair is None:
        cointegrated = [r for r in tester.results if r["is_cointegrated"]]
        best_pair = cointegrated[0] if cointegrated else tester.results[0]
        selection_mode = "auto"

    coint_bt = _run_strategy_backtest_for_pair(
        prices=prices,
        pair=best_pair["pair"],
        beta=float(best_pair["beta"]),
        spread=best_pair["spread"],
        z_window=z_window,
        entry_z=entry_z,
        exit_z=exit_z,
        max_holding_days=max_holding_days,
    )
    signals = coint_bt["signals"]
    coint_backtest_result = coint_bt.get("backtest_result")
    if coint_backtest_result is None:
        coint_backtest_result = {
            "returns": pd.Series(dtype=float),
            "equity_curve": pd.Series(dtype=float),
            "cumulative_returns": pd.Series(dtype=float),
            "drawdown": pd.Series(dtype=float),
            "metrics": {},
            "trades": pd.DataFrame(),
        }
    trades = coint_backtest_result["trades"]

    # Анализ динамики спреда (раздел 3.3)
    ticker_1, ticker_2 = best_pair["pair"]
    beta = float(best_pair["beta"])
    spread = calculate_spread(prices[ticker_1], prices[ticker_2], beta)
    pair_label = f"{ticker_1} - {ticker_2}"
    spread_df = calculate_zscore(spread=spread, window=z_window)
    spread_stats = spread_statistics(spread)
    saved_spread_files = save_spread_analysis(spread_df)
    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    spread_plot_path = plot_spread(spread, pair_label, out_dir / "spread_dynamics.png")
    zscore_plot_path = plot_zscore(
        spread_df["z_score"],
        pair_label,
        out_dir / "zscore_dynamics.png",
        entry_z=entry_z,
        exit_z=exit_z,
    )

    coint_details = coint_bt["details"]

    corr_analyzer = CorrelationAnalyzer(prices)
    corr_matrix = corr_analyzer.compute_correlation_matrix()
    corr_heatmap_path = _save_correlation_heatmap(corr_matrix)

    corr_pair = _get_top_correlation_pair(prices=prices)
    correlation_backtest = None
    best_method = "cointegration"
    comparison_reason = "Корреляционный бенчмарк недоступен для выбранной выборки."

    if corr_pair is not None:
        corr_bt = _run_strategy_backtest_for_pair(
            prices=prices,
            pair=corr_pair["pair"],
            beta=float(corr_pair["beta"]),
            spread=corr_pair["spread"],
            z_window=z_window,
            entry_z=entry_z,
            exit_z=exit_z,
            max_holding_days=max_holding_days,
        )
        correlation_backtest = {"pair": corr_pair, **corr_bt}

        coint_metrics = coint_backtest_result["metrics"]
        corr_metrics = corr_bt["backtest_result"]["metrics"]
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
        "metrics": coint_backtest_result["metrics"],
        "equity": coint_backtest_result["equity_curve"],
        "backtest_result": coint_backtest_result,
        "details": coint_details,
        "correlation_backtest": correlation_backtest,
        "comparison_table": tester.get_comparison_table(),
        "best_method": best_method,
        "comparison_reason": comparison_reason,
        "comparison_section": {
            "correlation_matrix": corr_matrix,
            "correlation_heatmap_path": corr_heatmap_path,
        },
        "cointegration_analysis": {
            "results_df": coint_results_df,
            "saved_files": coint_saved_files,
            "heatmap_path": coint_heatmap_path,
        },
        "spread_analysis": {
            "pair": best_pair["pair"],
            "beta": beta,
            "spread_df": spread_df,
            "spread_stats": spread_stats,
            "files": {
                **saved_spread_files,
                "spread_plot_png": spread_plot_path,
                "zscore_plot_png": zscore_plot_path,
            },
        },
        "selection_context": {
            "mode": selection_mode,
            "source": pair_source,
            "selected_pair": best_pair["pair"],
            "sample_start": str(prices.index.min().date()),
            "sample_end": str(prices.index.max().date()),
            "sample_size": int(len(prices)),
            "params": {
                "p_value_threshold": float(p_value_threshold),
                "z_window": int(z_window),
                "entry_z": float(entry_z),
                "exit_z": float(exit_z),
                "max_holding_days": int(max_holding_days),
            },
        },
    }
