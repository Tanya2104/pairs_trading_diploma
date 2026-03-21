"""
Поиск коинтегрированных пар методом Энгла-Грэнджера
Сравнение с корреляционным анализом
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict, Optional

from core.regression import LinearRegression
from core.adf_test import ADFTest
from core.correlation import CorrelationAnalyzer


class CointegrationTester:
    """
    Поиск коинтегрированных пар методом Энгла-Грэнджера
    """
    
    def __init__(self, prices: pd.DataFrame, p_value_threshold: float = 0.05):
        """
        Parameters
        ----------
        prices : pd.DataFrame
            Цены закрытия (индекс - дата, колонки - тикеры)
        p_value_threshold : float
            Порог p-value для отбора коинтегрированных пар
        """
        self.prices = prices
        self.threshold = p_value_threshold
        self.results = []
        self.correlation_analyzer = CorrelationAnalyzer(prices)
        self.adf_tester = ADFTest(max_lags=10, autolag='AIC')
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Расчёт периода полураспада спреда
        Half-life = -ln(2) / θ, где θ — коэффициент авторегрессии
        """
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        # Синхронизируем
        min_len = min(len(spread_lag), len(spread_diff))
        if min_len < 5:
            return np.inf
        
        spread_lag = spread_lag.iloc[:min_len]
        spread_diff = spread_diff.iloc[:min_len]
        
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(spread_lag, spread_diff)
            theta = slope
            
            if theta < 0:
                half_life = -np.log(2) / theta
            else:
                half_life = np.inf
        except:
            half_life = np.inf
        
        return half_life
    
    def test_pair(self, ticker1: str, ticker2: str) -> Optional[Dict]:
        """
        Проверка одной пары на коинтеграцию
        
        Returns
        -------
        dict or None
            Результаты анализа пары
        """
        x = self.prices[ticker1]
        y = self.prices[ticker2]
        
        # Убираем пропуски
        df_pair = pd.concat([x, y], axis=1).dropna()
        if len(df_pair) < 30:
            return None
        
        X = df_pair.iloc[:, 0]
        Y = df_pair.iloc[:, 1]
        
        # 1. Регрессия Y ~ X
        model = LinearRegression()
        model.fit(X, Y)
        coeffs = model.get_coefficients()
        residuals = model.get_residuals()
        
        # 2. ADF-тест остатков
        adf_result = self.adf_tester.run(pd.Series(residuals))
        
        # 3. Период полураспада
        spread = pd.Series(residuals, index=df_pair.index)
        half_life = self._calculate_half_life(spread)
        
        # 4. Корреляция
        correlation = self.correlation_analyzer.pearson_correlation(X, Y)
        
        result = {
            'pair': (ticker1, ticker2),
            'p_value': adf_result['p_value'],
            'adf_stat': adf_result['adf_stat'],
            'alpha': coeffs['alpha'],
            'beta': coeffs['beta'],
            'r_squared': coeffs['r_squared'],
            'spread': spread,
            'half_life': half_life,
            'correlation': correlation,
            'is_cointegrated': adf_result['is_stationary']
        }
        
        return result
    
    def find_pairs(self) -> List[Dict]:
        """
        Перебирает все пары и находит коинтегрированные
        """
        tickers = self.prices.columns.tolist()
        self.results = []
        
        total_pairs = len(list(combinations(tickers, 2)))
        print(f"Анализ {len(tickers)} акций, всего пар: {total_pairs}")
        
        for t1, t2 in combinations(tickers, 2):
            result = self.test_pair(t1, t2)
            if result:
                if result['is_cointegrated']:
                    print(f"  ✅ {t1}-{t2}: p={result['p_value']:.4f}, "
                          f"r²={result['r_squared']:.3f}, r={result['correlation']:.3f}")
                self.results.append(result)
        
        # Сортировка по p-value (коинтегрированные в начале)
        self.results.sort(key=lambda x: x['p_value'])
        
        return self.results
    
    def get_best_pair(self) -> Optional[Dict]:
        """Возвращает лучшую коинтегрированную пару"""
        if not self.results:
            self.find_pairs()
        
        cointegrated = [r for r in self.results if r['is_cointegrated']]
        if cointegrated:
            return cointegrated[0]
        return None
    
    def get_comparison_table(self) -> List[Dict]:
        """
        Возвращает таблицу сравнения корреляции и коинтеграции
        """
        if not self.results:
            self.find_pairs()
        
        return self.correlation_analyzer.compare_with_cointegration(self.results)


# ============= ТЕСТ =============
if __name__ == "__main__":
    from core.data_loader import MOEXLoader
    from core.data_processor import DataProcessor
    from config import data_config
    
    print("=" * 60)
    print("Поиск коинтегрированных пар (метод Энгла-Грэнджера)")
    print("=" * 60)
    
    # 1. Загружаем сырые данные
    loader = MOEXLoader(use_cache=True)
    raw_prices = loader.load_prices(
        tickers=data_config.tickers[:8],  # первые 8 для теста
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print(f"\nЗагружено сырых данных: {raw_prices.shape}")
    
    # 2. Обрабатываем данные
    print("\nОбработка данных...")
    processor = DataProcessor(raw_prices)
    processor.check_quality()
    processor.remove_empty_tickers(threshold=0.3)
    processor.synchronize_dates()
    
    prices = processor.get_processed_data()
    print(f"После обработки: {prices.shape}")
    
    # 3. Поиск коинтегрированных пар
    tester = CointegrationTester(prices, p_value_threshold=0.05)
    results = tester.find_pairs()
    
    # 4. Вывод результатов
    print(f"\n{'='*60}")
    print("Результаты поиска коинтегрированных пар")
    print(f"{'='*60}")
    
    cointegrated = [r for r in results if r['is_cointegrated']]
    print(f"\nНайдено коинтегрированных пар: {len(cointegrated)}")
    
    for r in cointegrated[:10]:
        print(f"\n  {r['pair'][0]} - {r['pair'][1]}:")
        print(f"    p-value: {r['p_value']:.6f}")
        print(f"    β: {r['beta']:.4f}")
        print(f"    R²: {r['r_squared']:.4f}")
        print(f"    Корреляция: {r['correlation']:.4f}")
        print(f"    Half-life: {r['half_life']:.1f} дней")
    
    # 5. Лучшая пара
    best = tester.get_best_pair()
    if best:
        print(f"\n{'='*60}")
        print(f"Лучшая пара: {best['pair'][0]} - {best['pair'][1]}")
        print(f"{'='*60}")
        print(f"  p-value: {best['p_value']:.6f}")
        print(f"  β: {best['beta']:.4f}")
        print(f"  α: {best['alpha']:.2f}")
        print(f"  R²: {best['r_squared']:.4f}")
        print(f"  Корреляция: {best['correlation']:.4f}")
        print(f"  Half-life: {best['half_life']:.1f} дней")
    
    # 6. Сравнительная таблица
    print(f"\n{'='*60}")
    print("Сравнение корреляции и коинтеграции")
    print(f"{'='*60}")
    
    comparison = tester.get_comparison_table()
    for row in comparison[:15]:
        p_coint = f"{row['p_value_coint']:.6f}" if row['p_value_coint'] else "None"
        print(f"  {row['pair']}: r={row['correlation']:.3f}, p_coint={p_coint}, {row['status']}")
    
    print(tester.correlation_analyzer.summary(comparison))