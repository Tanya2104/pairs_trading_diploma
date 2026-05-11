# -*- coding: utf-8 -*-
"""Cointegrated-pairs search via the Engle-Granger approach."""

from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adf_test import ADFTest
from core.correlation import CorrelationAnalyzer
from core.regression import LinearRegression


class CointegrationTester:
    """Find and rank potentially cointegrated ticker pairs."""

    def __init__(
        self,
        prices: pd.DataFrame,
        p_value_threshold: float = 0.05,
        min_r_squared: float = 0.6,
        min_half_life: float = 2.0,
        max_half_life: float = 120.0,
    ):
        self.prices = prices
        self.threshold = p_value_threshold
        self.min_r_squared = min_r_squared
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
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

    def _engle_granger_test(self, x: pd.Series, y: pd.Series, residuals: pd.Series) -> float:
        """Возвращает p-value теста Энгла–Грэнджера (statsmodels.coint)."""
        try:
            from statsmodels.tsa.stattools import coint

            _, p_value, _ = coint(y, x)
            return float(p_value)
        except (ImportError, ValueError, TypeError):
            # Fallback к ADF на остатках, если coint недоступен/неустойчив.
            adf_result = self.adf_tester.run(residuals)
            return float(adf_result["p_value"])

    def test_pair(self, ticker1: str, ticker2: str) -> Optional[Dict]:
        x = self.prices[ticker1]
        y = self.prices[ticker2]

        df_pair = pd.concat([x, y], axis=1).dropna()
        if len(df_pair) < 30:
            return None

        X = np.log(df_pair.iloc[:, 0])
        Y = np.log(df_pair.iloc[:, 1])

        model = LinearRegression()
        model.fit(X, Y)
        coeffs = model.get_coefficients()
        residuals = pd.Series(model.get_residuals(), index=df_pair.index)

        p_value = self._engle_granger_test(X, Y, residuals)
        adf_result = self.adf_tester.run(residuals)
        half_life = self._calculate_half_life(residuals)
        correlation = self.correlation_analyzer.pearson_correlation(X, Y)

        return {
            "pair": (ticker1, ticker2),
            "p_value": p_value,
            "adf_stat": adf_result["adf_stat"],
            "alpha": coeffs["alpha"],
            "beta": coeffs["beta"],
            "r_squared": coeffs["r_squared"],
            "spread": residuals,
            "half_life": half_life,
            "correlation": correlation,
            "cointegrated": p_value < self.threshold,
            "is_cointegrated": p_value < self.threshold,
        }

    def find_pairs(self) -> List[Dict]:
        tickers = self.prices.columns.tolist()
        self.results = []

        total_pairs = len(list(combinations(tickers, 2)))
        print(f"Analyzing {len(tickers)} assets, total pairs: {total_pairs}")

        for t1, t2 in combinations(tickers, 2):
            result = self.test_pair(t1, t2)
            if result:
                self.results.append(result)

        self.results.sort(key=lambda x: x["p_value"])
        return self.results

    def results_to_dataframe(self) -> pd.DataFrame:
        """Итоговая таблица (pair, p_value, beta, r_squared, half_life, cointegrated)."""
        if not self.results:
            self.find_pairs()

        rows = []
        for r in self.results:
            rows.append(
                {
                    "pair": f"{r['pair'][0]}-{r['pair'][1]}",
                    "p_value": float(r["p_value"]),
                    "beta": float(r["beta"]),
                    "r_squared": float(r["r_squared"]),
                    "half_life": float(r["half_life"]) if np.isfinite(r["half_life"]) else np.nan,
                    "cointegrated": bool(r["cointegrated"]),
                }
            )

        return pd.DataFrame(rows).sort_values("p_value", ascending=True).reset_index(drop=True)

    def save_results(self, output_dir: str = "data/results") -> Dict[str, str]:
        """Сохраняет полную таблицу и только коинтегрированные пары в CSV."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_df = self.results_to_dataframe()
        coint_df = results_df[results_df["cointegrated"]].copy()

        full_csv = output_path / "cointegration_results_all_pairs.csv"
        coint_csv = output_path / "cointegration_results_cointegrated_pairs.csv"

        results_df.to_csv(full_csv, index=False)
        coint_df.to_csv(coint_csv, index=False)

        return {"full_csv": str(full_csv), "cointegrated_csv": str(coint_csv)}

    def build_pvalue_matrix(self) -> pd.DataFrame:
        """Матрица p-value для heatmap."""
        if not self.results:
            self.find_pairs()

        tickers = self.prices.columns.tolist()
        matrix = pd.DataFrame(np.nan, index=tickers, columns=tickers)

        for t in tickers:
            matrix.loc[t, t] = 0.0

        for result in self.results:
            t1, t2 = result["pair"]
            matrix.loc[t1, t2] = result["p_value"]
            matrix.loc[t2, t1] = result["p_value"]

        return matrix

    def save_pvalue_heatmap(self, output_path: str = "data/results/heatmap_pvalue_cointegration.png") -> str:
        """Сохраняет heatmap матрицы p-value в PNG."""
        pval_matrix = self.build_pvalue_matrix()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pval_matrix,
            annot=True,
            fmt=".3f",
            cmap="viridis_r",
            linewidths=0.5,
            cbar_kws={"label": "p-value теста Энгла–Грэнджера"},
        )
        plt.title("Тепловая карта p-value коинтеграционных зависимостей")
        plt.xlabel("Тикер")
        plt.ylabel("Тикер")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return output_path

    def _is_tradable_pair(self, result: Dict) -> bool:
        if not result["is_cointegrated"]:
            return False
        beta = result["beta"]
        half_life = result["half_life"]
        r_squared = result["r_squared"]
        if beta <= 0:
            return False
        if not np.isfinite(half_life):
            return False
        if not (self.min_half_life <= half_life <= self.max_half_life):
            return False
        if r_squared < self.min_r_squared:
            return False
        return True

    def get_best_pair(self) -> Optional[Dict]:
        if not self.results:
            self.find_pairs()
        tradable = [r for r in self.results if self._is_tradable_pair(r)]
        if tradable:
            tradable.sort(key=lambda x: (x["p_value"], -x["r_squared"], x["half_life"]))
            return tradable[0]
        cointegrated = [r for r in self.results if r["is_cointegrated"]]
        return cointegrated[0] if cointegrated else None

    def get_comparison_table(self) -> List[Dict]:
        if not self.results:
            self.find_pairs()
        return self.correlation_analyzer.compare_with_cointegration(self.results)
