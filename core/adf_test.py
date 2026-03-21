"""
Расширенный тест Дики-Фуллера (ADF)
Обёртка над statsmodels для проверки стационарности
"""

import pandas as pd
from typing import Dict
from statsmodels.tsa.stattools import adfuller


class ADFTest:
    """
    Расширенный тест Дики-Фуллера
    """
    
    def __init__(self, max_lags: int = 10, autolag: str = 'AIC'):
        self.max_lags = max_lags
        self.autolag = autolag
    
    def run(self, series: pd.Series) -> Dict:
        """
        Выполнение ADF-теста
        
        Returns
        -------
        dict
            adf_stat, p_value, critical_values, used_lags, is_stationary
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            return {
                'adf_stat': 0,
                'p_value': 1.0,
                'critical_values': {},
                'used_lags': 0,
                'is_stationary': False
            }
        
        # Используем statsmodels
        result = adfuller(
            series_clean,
            maxlag=self.max_lags,
            autolag=self.autolag,
            regression='c'  # модель с константой
        )
        
        adf_stat = result[0]
        p_value = result[1]
        used_lags = result[2]
        critical_values = result[4]
        
        return {
            'adf_stat': adf_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'used_lags': used_lags,
            'is_stationary': p_value < 0.05
        }


# ============= ТЕСТ =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from core.data_loader import MOEXLoader
    
    loader = MOEXLoader(use_cache=True)
    prices = loader.load_prices(
        tickers=["SBER"],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print("=" * 60)
    print("Тест ADF (через statsmodels)")
    print("=" * 60)
    
    adf = ADFTest(max_lags=10, autolag='AIC')
    
    # Цены
    print("\n[1] Цены SBER")
    result = adf.run(prices['SBER'])
    print(f"  ADF: {result['adf_stat']:.4f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Стационарны: {result['is_stationary']}")
    
    # Доходности
    print("\n[2] Доходности SBER")
    returns = prices['SBER'].pct_change().dropna()
    result2 = adf.run(returns)
    print(f"  ADF: {result2['adf_stat']:.4f}")
    print(f"  p-value: {result2['p_value']:.4f}")
    print(f"  Стационарны: {result2['is_stationary']}")