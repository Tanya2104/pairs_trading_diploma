from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class PairsTradingStrategy:
    """Рыночно-нейтральная стратегия на основе спреда."""

    def __init__(self, spread: pd.Series, window: int, entry_z: float, exit_z: float):
        self.spread = spread
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.zscore: Optional[pd.Series] = None
        self.signals: Optional[pd.DataFrame] = None

    def calculate_zscore(self) -> pd.Series:
        """Расчёт Z-score на скользящем окне."""
        rolling_mean = self.spread.rolling(window=self.window).mean()
        rolling_std = self.spread.rolling(window=self.window).std()
        self.zscore = (self.spread - rolling_mean) / rolling_std
        return self.zscore

    def generate_signals(self, max_holding_days: int) -> pd.DataFrame:
        """Backward-compatible wrapper for trading signals generation."""
        self.signals = generate_trading_signals(
            spread=self.spread,
            rolling_window=self.window,
            entry_threshold=self.entry_z,
            exit_threshold=self.exit_z,
            max_holding_days=max_holding_days,
        )
        self.zscore = self.signals["zscore"]
        return self.signals

    def get_trades(self) -> pd.DataFrame:
        """Возвращает список сделок с датами входа и выхода."""
        if self.signals is None:
            raise ValueError("Signals are not generated. Call generate_signals() first.")
        return build_trades_table(self.signals)


def generate_trading_signals(
    spread: pd.Series,
    rolling_window: int,
    entry_threshold: float,
    exit_threshold: float,
    max_holding_days: int,
) -> pd.DataFrame:
    """Generate position state based on z-score thresholds and max holding period."""
    rolling_mean = spread.rolling(window=rolling_window).mean()
    rolling_std = spread.rolling(window=rolling_window).std()
    zscore = (spread - rolling_mean) / rolling_std

    signals = pd.DataFrame(index=spread.index)
    signals["spread"] = spread
    signals["zscore"] = zscore
    signals["position"] = 0
    signals["entry_flag"] = 0
    signals["exit_flag"] = 0
    signals["exit_reason"] = None

    position = 0
    days_held = 0

    for idx in signals.index:
        z = signals.at[idx, "zscore"]
        if pd.isna(z):
            position = 0
            days_held = 0
            continue

        if position == 0:
            if z > entry_threshold:
                position = -1
                days_held = 1
                signals.at[idx, "entry_flag"] = -1
            elif z < -entry_threshold:
                position = 1
                days_held = 1
                signals.at[idx, "entry_flag"] = 1
        else:
            days_held += 1
            if abs(z) < exit_threshold:
                signals.at[idx, "exit_flag"] = position
                signals.at[idx, "exit_reason"] = "возврат к среднему"
                position = 0
                days_held = 0
            elif days_held > max_holding_days:
                signals.at[idx, "exit_flag"] = position
                signals.at[idx, "exit_reason"] = "превышение максимального срока удержания"
                position = 0
                days_held = 0

        signals.at[idx, "position"] = position

    return signals


def build_trades_table(signals: pd.DataFrame) -> pd.DataFrame:
    """Build trades journal with entry/exit details and return per trade."""
    trades = []
    current_trade = None

    for dt, row in signals.iterrows():
        if row["entry_flag"] in (1, -1):
            current_trade = {
                "entry_date": dt,
                "position": "long spread" if row["entry_flag"] == 1 else "short spread",
                "entry_position": int(row["entry_flag"]),
                "entry_z": float(row["zscore"]),
                "entry_spread_raw": float(row["spread"]),
            }

        if current_trade is not None and row["exit_flag"] in (1, -1):
            exit_spread = float(row["spread"])
            entry_spread = float(current_trade["entry_spread_raw"])
            direction = int(current_trade["entry_position"])
            trade_return = direction * (exit_spread - entry_spread)

            trades.append(
                {
                    "entry_date": current_trade["entry_date"],
                    "exit_date": dt,
                    "position": current_trade["position"],
                    "entry_z": current_trade["entry_z"],
                    "exit_z": float(row["zscore"]),
                    "holding_days": int((dt - current_trade["entry_date"]).days),
                    "exit_reason": row["exit_reason"],
                    "pnl": trade_return,
                    "entry_spread_raw": entry_spread,
                    "exit_spread_raw": exit_spread,
                }
            )
            current_trade = None

    required_columns = [
        "entry_date",
        "exit_date",
        "position",
        "entry_z",
        "exit_z",
        "holding_days",
        "exit_reason",
        "pnl",
    ]
    extra_columns = ["entry_spread_raw", "exit_spread_raw"]
    return pd.DataFrame(trades, columns=required_columns + extra_columns)


def plot_zscore_signals(
    signals: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
) -> go.Figure:
    """Plot z-score with entry/exit levels and trade markers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=signals.index, y=signals["zscore"], mode="lines", name="Z-score"))

    for level, name, color, dash in [
        (entry_threshold, "+entry_threshold", "red", "dash"),
        (-entry_threshold, "-entry_threshold", "red", "dash"),
        (exit_threshold, "+exit_threshold", "green", "dot"),
        (-exit_threshold, "-exit_threshold", "green", "dot"),
        (0.0, "0", "gray", "solid"),
    ]:
        fig.add_hline(y=level, line_color=color, line_dash=dash, annotation_text=name)

    long_entries = signals[signals["entry_flag"] == 1]
    short_entries = signals[signals["entry_flag"] == -1]
    exits = signals[signals["exit_flag"].isin([1, -1])]

    fig.add_trace(go.Scatter(x=long_entries.index, y=long_entries["zscore"], mode="markers", name="Вход long", marker=dict(symbol="triangle-up", color="green", size=9)))
    fig.add_trace(go.Scatter(x=short_entries.index, y=short_entries["zscore"], mode="markers", name="Вход short", marker=dict(symbol="triangle-down", color="red", size=9)))
    fig.add_trace(go.Scatter(x=exits.index, y=exits["zscore"], mode="markers", name="Выход", marker=dict(symbol="x", color="black", size=9)))

    fig.update_layout(title="Торговые сигналы на основе Z-score", xaxis_title="Дата", yaxis_title="Z-score", template="plotly_white")
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def plot_spread_trades(signals: pd.DataFrame) -> go.Figure:
    """Plot spread with long/short entries and exits."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=signals.index, y=signals["spread"], mode="lines", name="Спред", line=dict(color="#1f77b4")))

    long_entries = signals[signals["entry_flag"] == 1]
    short_entries = signals[signals["entry_flag"] == -1]
    exits = signals[signals["exit_flag"].isin([1, -1])]

    fig.add_trace(go.Scatter(x=long_entries.index, y=long_entries["spread"], mode="markers", name="Открытие long spread", marker=dict(symbol="triangle-up", color="green", size=10)))
    fig.add_trace(go.Scatter(x=short_entries.index, y=short_entries["spread"], mode="markers", name="Открытие short spread", marker=dict(symbol="triangle-down", color="red", size=10)))
    fig.add_trace(go.Scatter(x=exits.index, y=exits["spread"], mode="markers", name="Закрытие позиции", marker=dict(symbol="x", color="black", size=9)))

    fig.update_layout(title="Сделки на графике спреда", xaxis_title="Дата", yaxis_title="Спред", template="plotly_white")
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig
