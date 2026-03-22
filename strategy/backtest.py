"""
Р‘СЌРєС‚РµСЃС‚РёСЂРѕРІР°РЅРёРµ С‚РѕСЂРіРѕРІРѕР№ СЃС‚СЂР°С‚РµРіРёРё
Р Р°СЃС‡С‘С‚ РјРµС‚СЂРёРє СЌС„С„РµРєС‚РёРІРЅРѕСЃС‚Рё
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict


class Backtest:
    """
    РњРѕРґРµР»РёСЂРѕРІР°РЅРёРµ С‚РѕСЂРіРѕРІРѕР№ СЃС‚СЂР°С‚РµРіРёРё
    """
    
    def __init__(
        self,
        signals: pd.DataFrame,
        spread: pd.Series,
        pair_returns: pd.Series | None = None,
        initial_capital: float = 1.0,
        volatility_scale: float = 1.0,
    ):
        """
        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame СЃ РєРѕР»РѕРЅРєРѕР№ 'position'
        spread : pd.Series
            Р’СЂРµРјРµРЅРЅРѕР№ СЂСЏРґ СЃРїСЂРµРґР°
        initial_capital : float
            РќР°С‡Р°Р»СЊРЅС‹Р№ РєР°РїРёС‚Р°Р» (РІ РѕС‚РЅРѕСЃРёС‚РµР»СЊРЅС‹С… РµРґРёРЅРёС†Р°С…)
        volatility_scale : float
            РњР°СЃС€С‚Р°Р±РёСЂСѓСЋС‰РёР№ РєРѕСЌС„С„РёС†РёРµРЅС‚ РґР»СЏ РїСЂРёРІРµРґРµРЅРёСЏ СЃРїСЂРµРґР° Рє СЃРѕРїРѕСЃС‚Р°РІРёРјРѕРјСѓ РјР°СЃС€С‚Р°Р±Сѓ
        """
        self.signals = signals
        self.spread = spread
        self.pair_returns = pair_returns
        self.initial_capital = initial_capital
        self.volatility_scale = volatility_scale
        self.returns = None
        self.cumulative_returns = None
        self.metrics = None

    def _get_base_returns(self) -> pd.Series:
        """
        Базовая доходность для пары.
        Приоритет: готовые доходности пары (r_y - beta * r_x), переданные из pipeline.
        Fallback: нормированное изменение спреда.
        """
        if self.pair_returns is not None:
            return self.pair_returns.reindex(self.signals.index).fillna(0.0)

        spread_diff = self.spread.diff().fillna(0.0)
        spread_scale = self.spread.rolling(window=20, min_periods=5).std().bfill()
        spread_scale = spread_scale.replace(0, np.nan).fillna(1.0)
        return spread_diff / spread_scale
    
    def run(self) -> Dict:
        """
        Р—Р°РїСѓСЃРє Р±СЌРєС‚РµСЃС‚Р°
        """
        position = self.signals['position'].shift(1).fillna(0)
        base_returns = self._get_base_returns()
        self.returns = position * base_returns * self.volatility_scale

        # Ограничиваем экстремальные значения (для устойчивости метрик)
        self.returns = np.clip(self.returns, -0.5, 0.5)

        # Геометрическое накопление капитала на процентных доходностях.
        self.cumulative_returns = self.initial_capital * (1 + self.returns).cumprod()

        # Дневной P&L строим на изменении спреда и позиции предыдущего дня.
        # ВАЖНО: это не "процент цены", а P&L в единицах спреда.
        position = self.signals['position'].shift(1).fillna(0)
        spread_diff = self.spread.diff().fillna(0)
        raw_pnl = position * spread_diff

        # Нормируем P&L на локальную "типичную величину спреда",
        # чтобы значения были сопоставимы и без экстремальных скачков.
        spread_scale = (
            self.spread.abs().rolling(window=20, min_periods=5).mean().bfill()
        )
        spread_scale = spread_scale.replace(0, np.nan).fillna(1.0)
        self.returns = (raw_pnl / spread_scale) * self.volatility_scale

        # Ограничиваем экстремальные значения (для устойчивости метрик)
        self.returns = np.clip(self.returns, -0.2, 0.2)

        # Капитал считаем аддитивно через накопленный P&L.
        # Это корректнее для спред-стратегии, чем геометрическое compounding.
        self.cumulative_returns = self.initial_capital + self.returns.cumsum()
        
        # Р Р°СЃС‡С‘С‚ РјРµС‚СЂРёРє
        self.metrics = self._calculate_metrics()
        
        return {
            'returns': self.returns,
            'cumulative_returns': self.cumulative_returns,
            'metrics': self.metrics
        }
    
    def _calculate_metrics(self) -> Dict:
        """Р Р°СЃС‡С‘С‚ РјРµС‚СЂРёРє СЌС„С„РµРєС‚РёРІРЅРѕСЃС‚Рё"""
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
        
        # РћР±С‰Р°СЏ РґРѕС…РѕРґРЅРѕСЃС‚СЊ
        total_return = self.cumulative_returns.iloc[-1] - 1
        
        # Годовая доходность (252 торговых дня)
        n_days = len(self.cumulative_returns)
        if n_days > 0 and total_return > -1:
            annual_return = (1 + total_return) ** (252 / n_days) - 1
        else:
            annual_return = 0
        # Годовая доходность: для аддитивного P&L используем среднедневной темп.
        annual_return = returns.mean() * 252
        
        # Sharpe ratio (РіРѕРґРѕРІРѕР№)
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)
        sharpe = mean_return / std_return if std_return != 0 and std_return < 100 else 0
        sharpe = np.clip(sharpe, -10, 10)  # РѕРіСЂР°РЅРёС‡РёРІР°РµРј
        
        # РњР°РєСЃРёРјР°Р»СЊРЅР°СЏ РїСЂРѕСЃР°РґРєР°
        peak = self.cumulative_returns.expanding().max()
        drawdown = (self.cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        if np.isinf(max_drawdown) or np.isnan(max_drawdown):
            max_drawdown = 0
        max_drawdown = max(max_drawdown, -1.0)
        
        # Win rate РїРѕ РґРЅСЏРј
        win_days = (returns > 0).sum()
        loss_days = (returns < 0).sum()
        win_rate = win_days / (win_days + loss_days) if (win_days + loss_days) > 0 else 0
        
        # РљРѕР»РёС‡РµСЃС‚РІРѕ СЃРґРµР»РѕРє (СЃРјРµРЅР° РїРѕР·РёС†РёРё)
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
        """Р’РѕР·РІСЂР°С‰Р°РµС‚ С‚РµРєСЃС‚РѕРІСѓСЋ СЃРІРѕРґРєСѓ СЂРµР·СѓР»СЊС‚Р°С‚РѕРІ"""
        if self.metrics is None:
            self.run()
        
        if np.isinf(self.cumulative_returns.iloc[-1]):
            return """
            вљ пёЏ РћС€РёР±РєР°: РїРµСЂРµРїРѕР»РЅРµРЅРёРµ РІ СЂР°СЃС‡С‘С‚Р°С…
            РџРѕРїСЂРѕР±СѓР№С‚Рµ РёР·РјРµРЅРёС‚СЊ РїР°СЂР°РјРµС‚СЂС‹ СЃС‚СЂР°С‚РµРіРёРё РёР»Рё РјР°СЃС€С‚Р°Р±РёСЂРѕРІР°РЅРёРµ
            """
        
        return f"""
        рџ“Љ Р РµР·СѓР»СЊС‚Р°С‚С‹ Р±СЌРєС‚РµСЃС‚Р°
        {'='*50}
        РќР°С‡Р°Р»СЊРЅС‹Р№ РєР°РїРёС‚Р°Р»: {self.initial_capital}
        РљРѕРЅРµС‡РЅС‹Р№ РєР°РїРёС‚Р°Р»: {self.cumulative_returns.iloc[-1]:.4f}
        
        Р”РѕС…РѕРґРЅРѕСЃС‚СЊ:
          РћР±С‰Р°СЏ: {self.metrics['total_return']:.2%}
          Р“РѕРґРѕРІР°СЏ: {self.metrics['annual_return']:.2%}
        
        Р РёСЃРє:
          Sharpe ratio: {self.metrics['sharpe_ratio']:.2f}
          РњР°РєСЃ. РїСЂРѕСЃР°РґРєР°: {self.metrics['max_drawdown']:.2%}
        
        РЎС‚Р°С‚РёСЃС‚РёРєР°:
          Р”РѕР»СЏ РїСЂРёР±С‹Р»СЊРЅС‹С… РґРЅРµР№: {self.metrics['win_rate']:.2%}
          РћС†РµРЅРєР° С‡РёСЃР»Р° СЃРґРµР»РѕРє: {self.metrics['num_trades']}
        """


# ============= РўР•РЎРў =============
if __name__ == "__main__":
    from core.data_loader import MOEXLoader
    from core.cointegration import CointegrationTester
    from strategy.signals import PairsTradingStrategy
    from config import data_config
    
    print("=" * 60)
    print("Р‘СЌРєС‚РµСЃС‚РёСЂРѕРІР°РЅРёРµ")
    print("=" * 60)
    
    # Р—Р°РіСЂСѓР¶Р°РµРј РґР°РЅРЅС‹Рµ
    loader = MOEXLoader(use_cache=True)
    prices = loader.load_prices(
        tickers=data_config.tickers[:8],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # РќР°С…РѕРґРёРј РєРѕРёРЅС‚РµРіСЂРёСЂРѕРІР°РЅРЅС‹Рµ РїР°СЂС‹
    tester = CointegrationTester(prices, p_value_threshold=0.05)
    results = tester.find_pairs()
    best = tester.get_best_pair()
    
    if best:
        print(f"\nР›СѓС‡С€Р°СЏ РїР°СЂР°: {best['pair'][0]} - {best['pair'][1]}")
        
        # Р“РµРЅРµСЂРёСЂСѓРµРј СЃРёРіРЅР°Р»С‹
        strategy = PairsTradingStrategy(
            spread=best['spread'],
            window=20,
            entry_z=2.0,
            exit_z=0.0
        )
        signals = strategy.generate_signals(max_holding_days=15)
        
        # Р‘СЌРєС‚РµСЃС‚
        backtest = Backtest(signals, best['spread'], initial_capital=1.0)
        results_bt = backtest.run()
        
        print(backtest.get_summary())
        
        # РџРѕРєР°Р·С‹РІР°РµРј СЃС‚Р°С‚РёСЃС‚РёРєСѓ
        print("\nРЎС‚Р°С‚РёСЃС‚РёРєР° РґРѕС…РѕРґРЅРѕСЃС‚РµР№:")
        returns = results_bt['returns'].dropna()
        print(f"  РЎСЂРµРґРЅСЏСЏ: {returns.mean():.6f}")
        print(f"  Std: {returns.std():.6f}")
        print(f"  Min: {returns.min():.6f}")
        print(f"  Max: {returns.max():.6f}")
        
        print("\nРџРѕСЃР»РµРґРЅРёРµ 10 РґРЅРµР№ РєР°РїРёС‚Р°Р»Р°:")
        print(results_bt['cumulative_returns'].tail(10))
    else:
        print("РљРѕРёРЅС‚РµРіСЂРёСЂРѕРІР°РЅРЅС‹Рµ РїР°СЂС‹ РЅРµ РЅР°Р№РґРµРЅС‹")
