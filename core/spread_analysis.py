"""Инструменты анализа динамики спреда и Z-score для раздела 3.3."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def calculate_spread(price_1: pd.Series, price_2: pd.Series, beta: float) -> pd.Series:
    """Вычисляет спред: spread = price_1 - beta * price_2."""
    aligned = pd.concat([price_1, price_2], axis=1).dropna()
    spread = aligned.iloc[:, 0] - beta * aligned.iloc[:, 1]
    spread.name = "spread"
    return spread


def calculate_zscore(spread: pd.Series, window: int) -> pd.DataFrame:
    """Вычисляет rolling-mean/std и Z-score для заданного окна."""
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    z_score = (spread - rolling_mean) / rolling_std
    return pd.DataFrame({"spread": spread, "rolling_mean": rolling_mean, "rolling_std": rolling_std, "z_score": z_score})


def spread_statistics(spread: pd.Series) -> Dict[str, float]:
    """Базовая статистика по спреду."""
    return {
        "mean": float(spread.mean()),
        "std": float(spread.std()),
        "min": float(spread.min()),
        "max": float(spread.max()),
    }


def plot_spread(spread: pd.Series, pair_label: str, output_path: Path) -> str:
    """Строит и сохраняет график спреда с линией среднего."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(spread.index, spread.values, label="Спред", color="#1f77b4")
    ax.axhline(spread.mean(), color="#d62728", linestyle="--", label="Среднее значение")
    ax.set_title(f"Динамика спреда для пары {pair_label}")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Значение спреда")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def plot_zscore(
    zscore: pd.Series,
    pair_label: str,
    output_path: Path,
    entry_z: float,
    exit_z: float,
) -> str:
    """Строит и сохраняет график Z-score с актуальными пороговыми уровнями стратегии."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(zscore.index, zscore.values, label="Z-score", color="#ff7f0e")
    for level, style, color in [
        (entry_z, "--", "#d62728"),
        (-entry_z, "--", "#d62728"),
        (0, "-", "#2ca02c"),
        (exit_z, ":", "#9467bd"),
        (-exit_z, ":", "#9467bd"),
    ]:
        ax.axhline(level, linestyle=style, color=color, linewidth=1, label=f"Уровень {level:+g}")
    ax.set_title(f"Динамика Z-score для пары {pair_label}")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Z-score")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def save_spread_analysis(analysis_df: pd.DataFrame, output_dir: str = "data/results") -> Dict[str, str]:
    """Сохраняет таблицу spread/z-score в CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "spread_zscore_analysis.csv"
    analysis_df.to_csv(csv_path, index_label="Дата")
    return {"spread_zscore_csv": str(csv_path)}
