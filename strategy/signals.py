"""
Генерация торговых сигналов на основе Z-score спреда
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, Optional


class PairsTradingStrategy:
    """
    Рыночно-нейтральная стратегия на основе спреда
    """
    
    def __init__(self, spread: pd.Series, window: int = 20, 
                 entry_z: float = 2.0, exit_z: float = 0.0):
        """
        Parameters
        ----------
        spread : pd.Series
            Временной ряд спреда (остатки коинтеграционной регрессии)
        window : int
            Окно для расчёта скользящего среднего и стандартного отклонения
        entry_z : float
            Порог Z-score для открытия позиции
        exit_z : float
            Порог Z-score для закрытия позиции
        """
        self.spread = spread
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.zscore = None
        self.signals = None
    
    def calculate_zscore(self) -> pd.Series:
        """
        Расчёт Z-score на скользящем окне
        Z_t = (S_t - mu_t) / sigma_t
        
        Returns
        -------
        pd.Series
            Временной ряд Z-score
        """
        rolling_mean = self.spread.rolling(window=self.window).mean()
        rolling_std = self.spread.rolling(window=self.window).std()
        
        self.zscore = (self.spread - rolling_mean) / rolling_std
        return self.zscore
    
    def generate_signals(self, max_holding_days: int = 30) -> pd.DataFrame:
        """
        Генерация торговых сигналов с таймаутом для закрытия
        
        Правила:
        - Z < -entry_z -> длинная позиция (+1)
        - Z > entry_z -> короткая позиция (-1)
        - |Z| < exit_z -> закрыть позицию
        - Если позиция держится > max_holding_days -> принудительное закрытие
        
        Parameters
        ----------
        max_holding_days : int
            Максимальное количество дней удержания позиции
            
        Returns
        -------
        pd.DataFrame
            С колонками: zscore, signal, position
        """
        if self.zscore is None:
            self.calculate_zscore()
        
        signals = pd.DataFrame(index=self.zscore.index)
        signals['zscore'] = self.zscore
        signals['signal'] = 0
        
        # Открытие позиций
        signals.loc[self.zscore < -self.entry_z, 'signal'] = 1   # long
        signals.loc[self.zscore > self.entry_z, 'signal'] = -1   # short
        
        # Удержание позиции (ffill вместо replace с method)
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Закрытие при возврате к exit_z
        signals.loc[abs(self.zscore) < self.exit_z, 'position'] = 0
        
        # Закрытие по таймауту (если позиция держится слишком долго)
        days_held = 0
        position_values = signals['position'].values.copy()
        
        for i in range(len(position_values)):
            if position_values[i] != 0:
                days_held += 1
                if days_held >= max_holding_days:
                    position_values[i] = 0
                    days_held = 0
            else:
                days_held = 0
        
        signals['position'] = position_values
        
        self.signals = signals
        return signals
    
    def get_trades(self) -> pd.DataFrame:
        """
        Возвращает список сделок с датами входа/выхода
        
        Returns
        -------
        pd.DataFrame
            С колонками: entry_date, exit_date, direction, entry_z, exit_z, pnl
        """
        if self.signals is None:
            self.generate_signals()
        
        trades = []
        position = 0
        entry_date = None
        entry_z = None
        entry_spread = None
        
        for date, row in self.signals.iterrows():
            new_position = row['position']
            
            # Открытие позиции
            if position == 0 and new_position != 0:
                entry_date = date
                entry_z = row['zscore']
                entry_spread = self.spread.loc[date]
                position = new_position
            
            # Закрытие позиции
            elif position != 0 and new_position == 0:
                exit_date = date
                exit_z = row['zscore']
                exit_spread = self.spread.loc[date]
                
                # P&L: изменение спреда, умноженное на направление
                pnl = position * (exit_spread - entry_spread)
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'direction': 'long' if position == 1 else 'short',
                    'entry_z': entry_z,
                    'exit_z': exit_z,
                    'entry_spread': entry_spread,
                    'exit_spread': exit_spread,
                    'holding_days': (exit_date - entry_date).days,
                    'pnl': pnl
                })
                position = 0
        
        return pd.DataFrame(trades)
    
    def get_summary(self) -> str:
        """
        Возвращает текстовую сводку по сигналам
        """
        if self.signals is None:
            self.generate_signals()
        
        trades = self.get_trades()
        
        if len(trades) == 0:
            return "Нет сделок"
        
        long_trades = trades[trades['direction'] == 'long']
        short_trades = trades[trades['direction'] == 'short']
        profitable = len(trades[trades['pnl'] > 0])
        win_rate = profitable / len(trades) * 100 if len(trades) > 0 else 0
        
        return f"""
        📊 Сводка по сигналам
        {'='*40}
        Период: {self.signals.index[0].date()} - {self.signals.index[-1].date()}
        Параметры: окно={self.window}, entry_z={self.entry_z}, exit_z={self.exit_z}
        
        Сделки:
          Всего: {len(trades)}
          Длинные: {len(long_trades)}
          Короткие: {len(short_trades)}
          Прибыльных: {profitable} ({win_rate:.1f}%)
          Убыточных: {len(trades) - profitable}
        
        P&L:
          Общий: {trades['pnl'].sum():.4f}
          Средний: {trades['pnl'].mean():.4f}
          Макс. прибыль: {trades['pnl'].max():.4f}
          Макс. убыток: {trades['pnl'].min():.4f}
        """


# ============= ТЕСТ =============
if __name__ == "__main__":
    from core.data_loader import MOEXLoader
    from core.cointegration import CointegrationTester
    from config import data_config
    
    print("=" * 60)
    print("Тест генерации сигналов")
    print("=" * 60)
    
    # Загружаем данные
    loader = MOEXLoader(use_cache=True)
    prices = loader.load_prices(
        tickers=data_config.tickers[:8],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Находим коинтегрированные пары
    tester = CointegrationTester(prices, p_value_threshold=0.05)
    results = tester.find_pairs()
    best = tester.get_best_pair()
    
    if best:
        print(f"\nЛучшая пара: {best['pair'][0]} - {best['pair'][1]}")
        
        # Генерируем сигналы с таймаутом 15 дней
        strategy = PairsTradingStrategy(
            spread=best['spread'],
            window=20,
            entry_z=2.0,
            exit_z=0.0
        )
        
        signals = strategy.generate_signals(max_holding_days=15)
        
        # Показываем моменты, когда есть позиция
        print("\nСигналы (где position != 0):")
        non_zero = signals[signals['position'] != 0]
        if len(non_zero) > 0:
            print(non_zero.head(30))
        else:
            print("Нет ненулевых позиций")
            print("\nПервые 30 строк сигналов:")
            print(signals.head(30))
        
        print(strategy.get_summary())
        
        trades = strategy.get_trades()
        if len(trades) > 0:
            print("\nСделки:")
            print(trades)
        else:
            print("\nНет сделок (возможно, спред не превышал порог 2.0)")
    else:
        print("Коинтегрированные пары не найдены")