# -*- coding: utf-8 -*-
"""Cointegrated-pairs search via the Engle-Granger approach."""

from __future__ import annotations

import os
import sys
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adf_test import ADFTest
from core.correlation import CorrelationAnalyzer
from core.regression import LinearRegression


class CointegrationTester:
    """Find and rank potentially cointegrated ticker pairs."""

    def __init__(self, prices: pd.DataFrame, p_value_threshold: float = 0.05):
        self.prices = prices
        self.threshold = p_value_threshold
        self.results: List[Dict] = []
        self.correlation_analyzer = CorrelationAnalyzer(prices)
        self.adf_tester = ADFTest(max_lags=10, autolag="AIC")

    def _calculate_half_life(self, spread: pd.Series) -> float:
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        min_len = min(len(spread_lag), len(spread_diff))
        if min_len < 5:
            return np.inf

        spread_lag = spread_lag.iloc[:min_len]
        spread_diff = spread_diff.iloc[:min_len]

        try:
            from scipy import stats

            slope, _, _, _, _ = stats.linregress(spread_lag, spread_diff)
            theta = slope
            return -np.log(2) / theta if theta < 0 else np.inf
        except (ValueError, TypeError, ZeroDivisionError, ImportError):
            return np.inf

    def test_pair(self, ticker1: str, ticker2: str) -> Optional[Dict]:
        x = self.prices[ticker1]
        y = self.prices[ticker2]

        df_pair = pd.concat([x, y], axis=1).dropna()
        if len(df_pair) < 30:
            return None

        X = df_pair.iloc[:, 0]
        Y = df_pair.iloc[:, 1]

        model = LinearRegression()
        model.fit(X, Y)
        coeffs = model.get_coefficients()
        residuals = model.get_residuals()

        adf_result = self.adf_tester.run(pd.Series(residuals))
        spread = pd.Series(residuals, index=df_pair.index)
        half_life = self._calculate_half_life(spread)
        correlation = self.correlation_analyzer.pearson_correlation(X, Y)

        return {
            "pair": (ticker1, ticker2),
            "p_value": adf_result["p_value"],
            "adf_stat": adf_result["adf_stat"],
            "alpha": coeffs["alpha"],
            "beta": coeffs["beta"],
            "r_squared": coeffs["r_squared"],
            "spread": spread,
            "half_life": half_life,
            "correlation": correlation,
            "is_cointegrated": adf_result["is_stationary"] and adf_result["p_value"] < self.threshold,
        }

    def find_pairs(self) -> List[Dict]:
        tickers = self.prices.columns.tolist()
        self.results = []

        total_pairs = len(list(combinations(tickers, 2)))
        print(f"Analyzing {len(tickers)} assets, total pairs: {total_pairs}")

        for t1, t2 in combinations(tickers, 2):
            result = self.test_pair(t1, t2)
            if result:
                if result["is_cointegrated"]:
                    print(
                        f"  ✓ {t1}-{t2}: p={result['p_value']:.4f}, "
                        f"r²={result['r_squared']:.3f}, r={result['correlation']:.3f}"
                    )
                self.results.append(result)

        self.results.sort(key=lambda x: x["p_value"])
        return self.results

    def get_best_pair(self) -> Optional[Dict]:
        if not self.results:
            self.find_pairs()

        cointegrated = [r for r in self.results if r["is_cointegrated"]]
        return cointegrated[0] if cointegrated else None

    def get_comparison_table(self) -> List[Dict]:
        if not self.results:
            self.find_pairs()

        return self.correlation_analyzer.compare_with_cointegration(self.results)


if __name__ == "__main__":
    from config import data_config
    from core.data_loader import MOEXLoader
    from core.data_processor import DataProcessor

    print("=" * 60)
    print("Cointegrated-pairs search (Engle-Granger)")
    print("=" * 60)

    loader = MOEXLoader(use_cache=True)
    raw_prices = loader.load_prices(
        tickers=data_config.tickers[:8],
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    print(f"\nRaw data shape: {raw_prices.shape}")

    print("\nData preprocessing...")
    processor = DataProcessor(raw_prices)
    processor.check_quality()
    processor.remove_empty_tickers(threshold=0.3)
    processor.synchronize_dates()

    prices = processor.get_processed_data()
    print(f"After preprocessing: {prices.shape}")

    tester = CointegrationTester(prices, p_value_threshold=0.05)
    results = tester.find_pairs()

    print(f"\n{'=' * 60}")
    print("Cointegration scan results")
    print(f"{'=' * 60}")

    cointegrated = [r for r in results if r["is_cointegrated"]]
    print(f"\nCointegrated pairs found: {len(cointegrated)}")

    for r in cointegrated[:10]:
        print(f"\n  {r['pair'][0]} - {r['pair'][1]}:")
        print(f"    p-value: {r['p_value']:.6f}")
        print(f"    beta: {r['beta']:.4f}")
        print(f"    R²: {r['r_squared']:.4f}")
        print(f"    correlation: {r['correlation']:.4f}")
        print(f"    half-life: {r['half_life']:.1f} days")

    best = tester.get_best_pair()
    if best:
        print(f"\n{'=' * 60}")
        print(f"Best pair: {best['pair'][0]} - {best['pair'][1]}")
        print(f"{'=' * 60}")
        print(f"  p-value: {best['p_value']:.6f}")
        print(f"  beta: {best['beta']:.4f}")
        print(f"  alpha: {best['alpha']:.2f}")
        print(f"  R²: {best['r_squared']:.4f}")
        print(f"  correlation: {best['correlation']:.4f}")
        print(f"  half-life: {best['half_life']:.1f} days")

    print(f"\n{'=' * 60}")
    print("Correlation vs cointegration")
    print(f"{'=' * 60}")

    comparison = tester.get_comparison_table()
    for row in comparison[:15]:
        p_coint = f"{row['p_value_coint']:.6f}" if row["p_value_coint"] else "None"
        print(f"  {row['pair']}: r={row['correlation']:.3f}, p_coint={p_coint}, {row['status']}")

    print(tester.correlation_analyzer.summary(comparison))
