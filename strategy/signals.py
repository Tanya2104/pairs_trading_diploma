"""Генерация торговых сигналов на основе Z-score спреда."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


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
        """Генерация торговых сигналов с ограничением срока удержания позиции."""
        if self.zscore is None:
            self.calculate_zscore()

        signals = pd.DataFrame(index=self.zscore.index)
        signals["zscore"] = self.zscore
        signals["signal"] = 0

        signals.loc[self.zscore < -self.entry_z, "signal"] = 1
        signals.loc[self.zscore > self.entry_z, "signal"] = -1

        position = 0
        days_held = 0
        position_values = []

        for z, signal in zip(signals["zscore"].values, signals["signal"].values):
            if np.isnan(z):
                position_values.append(0)
                position = 0
                days_held = 0
                continue

            if position == 0:
                if signal != 0:
                    position = int(signal)
                    days_held = 1
                position_values.append(position)
                continue

            if abs(z) < self.exit_z:
                position = 0
                days_held = 0
                position_values.append(0)
                continue

            if signal != 0 and signal != position:
                position = int(signal)
                days_held = 1
                position_values.append(position)
                continue

            days_held += 1
            if days_held >= max_holding_days:
                position = 0
                days_held = 0
                position_values.append(0)
            else:
                position_values.append(position)

        signals["position"] = position_values
        self.signals = signals
        return signals

    def get_trades(self) -> pd.DataFrame:
        """Возвращает список сделок с датами входа и выхода."""
        if self.signals is None:
            raise ValueError("Signals are not generated. Call generate_signals() first.")

        trades = []
        position = 0
        entry_date = None
        entry_z = None
        entry_spread = None

        for date, row in self.signals.iterrows():
            new_position = row["position"]

            if position == 0 and new_position != 0:
                entry_date = date
                entry_z = row["zscore"]
                entry_spread = self.spread.loc[date]
                position = new_position

            elif position != 0 and new_position == 0:
                exit_date = date
                exit_z = row["zscore"]
                exit_spread = self.spread.loc[date]
                pnl = position * (exit_spread - entry_spread)

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "direction": "long" if position == 1 else "short",
                        "entry_z": entry_z,
                        "exit_z": exit_z,
                        "entry_spread": entry_spread,
                        "exit_spread": exit_spread,
                        "holding_days": (exit_date - entry_date).days,
                        "pnl": pnl,
                    }
                )
                position = 0

        return pd.DataFrame(trades)
