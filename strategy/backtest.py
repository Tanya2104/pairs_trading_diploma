"""
Бэктестирование торговой стратегии
Расчёт метрик эффективности
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict


class Backtest:
    """
    Моделирование торговой стратегии
    """
    
    def __init__(self, signals: pd.DataFrame, spread: pd.Series, 
                 initial_capital: float = 1.0, volatility_scale: float = 1.0):
        """
        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame с колонкой 'position'
        spread : pd.Series
            Временной ряд спреда
        initial_capital : float
            Начальный капитал (в относительных единицах)
        volatility_scale : float
            Масштабирующий коэффициент для приведения спреда к сопоставимому масштабу
        """
        self.signals = signals
        self.spread = spread
        self.initial_capital = initial_capital
        self.volatility_scale = volatility_scale
        self.returns = None
        self.cumulative_returns = None
        self.metrics = None
    
    def run(self) -> Dict:
        """
        Запуск бэктеста
        """
        # Нормализуем спред (приводим к сопоставимому масштабу)
        spread_normalized = self.spread / self.spread.std()
        
        # Доходность: изменение нормированного спреда, умноженное на позицию
        spread_returns = spread_normalized.diff().shift(-1)
        self.returns = self.signals['position'].shift(1) * spread_returns
        self.returns = self.returns.fillna(0)
        
        # Ограничиваем экстремальные значения (для устойчивости)
        self.returns = np.clip(self.returns, -0.5, 0.5)
        
        # Накопленная доходность
        self.cumulative_returns = (1 + self.returns).cumprod()
        self.cumulative_returns = self.cumulative_returns * self.initial_capital
        
        # Расчёт метрик
        self.metrics = self._calculate_metrics()
        
        return {
            'returns': self.returns,
            'cumulative_returns': self.cumulative_returns,
            'metrics': self.metrics
        }
    
    def _calculate_metrics(self) -> Dict:
        """Расчёт метрик эффективности"""
        returns = self.returns.dropna()
        
        if len(returns) == 0 or np.isinf(self.cumulative_returns.iloc[-1]):
            return {
                'total_return': 0,
                'annual_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'num_trades': 0
            }
        
        # Общая доходность
        total_return = self.cumulative_returns.iloc[-1] - 1
        
        # Годовая доходность (252 торговых дня)
        n_days = len(self.cumulative_returns)
        if n_days > 0 and total_return > -1:
            annual_return = (1 + total_return) ** (252 / n_days) - 1
        else:
            annual_return = 0
        
        # Sharpe ratio (годовой)
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)
        sharpe = mean_return / std_return if std_return != 0 and std_return < 100 else 0
        sharpe = np.clip(sharpe, -10, 10)  # ограничиваем
        
        # Максимальная просадка
        peak = self.cumulative_returns.expanding().max()
        drawdown = (self.cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        if np.isinf(max_drawdown) or np.isnan(max_drawdown):
            max_drawdown = 0
        
        # Win rate по дням
        win_days = (returns > 0).sum()
        loss_days = (returns < 0).sum()
        win_rate = win_days / (win_days + loss_days) if (win_days + loss_days) > 0 else 0
        
        # Количество сделок (смена позиции)
        position_changes = self.signals['position'].diff()
        trades = position_changes[position_changes != 0]
        num_trades = len(trades) // 2
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades
        }
    
    def get_summary(self) -> str:
        """Возвращает текстовую сводку результатов"""
        if self.metrics is None:
            self.run()
        
        if np.isinf(self.cumulative_returns.iloc[-1]):
            return """
            ⚠️ Ошибка: переполнение в расчётах
            Попробуйте изменить параметры стратегии или масштабирование
            """
        
        return f"""
        📊 Результаты бэктеста
        {'='*50}
        Начальный капитал: {self.initial_capital}
        Конечный капитал: {self.cumulative_returns.iloc[-1]:.4f}
        
        Доходность:
          Общая: {self.metrics['total_return']:.2%}
          Годовая: {self.metrics['annual_return']:.2%}
        
        Риск:
          Sharpe ratio: {self.metrics['sharpe_ratio']:.2f}
          Макс. просадка: {self.metrics['max_drawdown']:.2%}
        
        Статистика:
          Доля прибыльных дней: {self.metrics['win_rate']:.2%}
          Оценка числа сделок: {self.metrics['num_trades']}
        """


# ============= ТЕСТ =============
if __name__ == "__main__":
    from core.data_loader import MOEXLoader
    from core.cointegration import CointegrationTester
    from strategy.signals import PairsTradingStrategy
    from config import data_config
    
    print("=" * 60)
    print("Бэктестирование")
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
        
        # Генерируем сигналы
        strategy = PairsTradingStrategy(
            spread=best['spread'],
            window=20,
            entry_z=2.0,
            exit_z=0.0
        )
        signals = strategy.generate_signals(max_holding_days=15)
        
        # Бэктест
        backtest = Backtest(signals, best['spread'], initial_capital=1.0)
        results_bt = backtest.run()
        
        print(backtest.get_summary())
        
        # Показываем статистику
        print("\nСтатистика доходностей:")
        returns = results_bt['returns'].dropna()
        print(f"  Средняя: {returns.mean():.6f}")
        print(f"  Std: {returns.std():.6f}")
        print(f"  Min: {returns.min():.6f}")
        print(f"  Max: {returns.max():.6f}")
        
        print("\nПоследние 10 дней капитала:")
        print(results_bt['cumulative_returns'].tail(10))
    else:
        print("Коинтегрированные пары не найдены")