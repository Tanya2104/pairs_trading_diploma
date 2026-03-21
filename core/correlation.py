"""
Ручная реализация корреляции Пирсона
Для сравнения с коинтеграцией
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class CorrelationAnalyzer:
    """
    Анализ корреляционных связей между активами
    """
    
    def __init__(self, prices: pd.DataFrame):
        """
        Parameters
        ----------
        prices : pd.DataFrame
            Цены закрытия (индекс - дата, колонки - тикеры)
        """
        self.prices = prices
        self.correlation_matrix = None
    
    def pearson_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """
        Ручная реализация корреляции Пирсона
        
        r = Σ((x - x̄)(y - ȳ)) / √(Σ(x - x̄)² * Σ(y - ȳ)²)
        """
        # Убираем пропуски
        mask = ~(x.isna() | y.isna())
        x_clean = x[mask].values
        y_clean = y[mask].values
        
        n = len(x_clean)
        if n < 3:
            return 0
        
        # Средние
        mean_x = np.mean(x_clean)
        mean_y = np.mean(y_clean)
        
        # Отклонения
        x_dev = x_clean - mean_x
        y_dev = y_clean - mean_y
        
        # Корреляция
        numerator = np.sum(x_dev * y_dev)
        denominator = np.sqrt(np.sum(x_dev ** 2) * np.sum(y_dev ** 2))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def compute_correlation_matrix(self) -> pd.DataFrame:
        """
        Вычисляет матрицу корреляций для всех активов
        """
        tickers = self.prices.columns
        n = len(tickers)
        
        corr_matrix = pd.DataFrame(
            np.zeros((n, n)),
            index=tickers,
            columns=tickers
        )
        
        for i, t1 in enumerate(tickers):
            for j, t2 in enumerate(tickers):
                if i <= j:
                    corr = self.pearson_correlation(self.prices[t1], self.prices[t2])
                    corr_matrix.loc[t1, t2] = corr
                    corr_matrix.loc[t2, t1] = corr
        
        self.correlation_matrix = corr_matrix
        return corr_matrix
    
    def get_high_correlation_pairs(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Находит пары с корреляцией выше порога
        
        Returns
        -------
        list of (ticker1, ticker2, correlation)
        """
        if self.correlation_matrix is None:
            self.compute_correlation_matrix()
        
        pairs = []
        tickers = self.correlation_matrix.columns
        
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i+1:]:
                corr = self.correlation_matrix.loc[t1, t2]
                if abs(corr) > threshold:
                    pairs.append((t1, t2, corr))
        
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    
    def compare_with_cointegration(self, coint_results: List[Dict]) -> List[Dict]:
        """
        Сравнивает корреляцию с коинтеграцией
        
        Parameters
        ----------
        coint_results : list
            Результаты из CointegrationTester (список словарей)
            
        Returns
        -------
        list
            Сравнительная таблица
        """
        # Создаём словарь для быстрого доступа к результатам коинтеграции
        coint_dict = {}
        for r in coint_results:
            pair = r['pair']
            coint_dict[f"{pair[0]}-{pair[1]}"] = r
        
        comparison = []
        tickers = self.prices.columns
        
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i+1:]:
                pair_name = f"{t1}-{t2}"
                corr = self.pearson_correlation(self.prices[t1], self.prices[t2])
                
                coint_info = coint_dict.get(pair_name)
                
                if coint_info:
                    p_value = coint_info['p_value']
                    is_cointegrated = p_value < 0.05
                    beta = coint_info.get('beta', None)
                    half_life = coint_info.get('half_life', None)
                else:
                    p_value = None
                    is_cointegrated = False
                    beta = None
                    half_life = None
                
                # Определяем статус
                if is_cointegrated and abs(corr) > 0.7:
                    status = "✅ Коинтеграция + высокая корреляция"
                elif is_cointegrated and abs(corr) <= 0.7:
                    status = "🔍 Скрытая коинтеграция (корреляция низкая)"
                elif not is_cointegrated and abs(corr) > 0.7:
                    status = "⚠️ Ложная корреляция (нет коинтеграции)"
                else:
                    status = "❌ Нет значимой связи"
                
                comparison.append({
                    'pair': pair_name,
                    'correlation': round(corr, 4),
                    'p_value_coint': round(p_value, 6) if p_value else None,
                    'is_cointegrated': is_cointegrated,
                    'beta': round(beta, 4) if beta else None,
                    'half_life': round(half_life, 1) if half_life else None,
                    'status': status
                })
        
        # Сортируем по абсолютной корреляции
        return sorted(comparison, key=lambda x: abs(x['correlation']), reverse=True)
    
    def summary(self, comparison: List[Dict]) -> str:
        """
        Возвращает текстовую сводку сравнения
        """
        high_corr = [c for c in comparison if abs(c['correlation']) > 0.7]
        coint_pairs = [c for c in comparison if c['is_cointegrated']]
        false_corr = [c for c in high_corr if not c['is_cointegrated']]
        hidden_coint = [c for c in coint_pairs if abs(c['correlation']) <= 0.7]
        
        return f"""
        📊 Сравнение корреляции и коинтеграции
        {'='*50}
        Всего пар: {len(comparison)}
        
        Высокая корреляция (>0.7): {len(high_corr)} пар
          Из них без коинтеграции: {len(false_corr)} пар
        
        Коинтегрированные пары: {len(coint_pairs)} пар
          Из них с низкой корреляцией: {len(hidden_coint)} пар
        
        Вывод: Корреляция не является надёжным критерием для парного трейдинга.
        """


# ============= ТЕСТ =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from core.data_loader import MOEXLoader
    from config import data_config
    
    # Загружаем данные
    loader = MOEXLoader(use_cache=True)
    prices = loader.load_prices(
        tickers=data_config.tickers[:5],  # первые 5 для теста
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print("=" * 60)
    print("Корреляционный анализ")
    print("=" * 60)
    
    # Анализируем корреляцию
    analyzer = CorrelationAnalyzer(prices)
    corr_matrix = analyzer.compute_correlation_matrix()
    
    print("\nМатрица корреляций:")
    print(corr_matrix.round(3))
    
    print("\nПары с высокой корреляцией (>0.8):")
    high_corr = analyzer.get_high_correlation_pairs(threshold=0.8)
    for t1, t2, corr in high_corr[:10]:
        print(f"  {t1} - {t2}: {corr:.4f}")
    
    # Имитируем результаты коинтеграции для демонстрации сравнения
    # (позже здесь будут реальные результаты из CointegrationTester)
    mock_coint_results = [
        {'pair': ('SBER', 'VTBR'), 'p_value': 0.03, 'beta': 1.2, 'half_life': 15},
        {'pair': ('GAZP', 'NVTK'), 'p_value': 0.01, 'beta': 0.8, 'half_life': 12},
    ]
    
    print("\nСравнение с коинтеграцией (на примере):")
    comparison = analyzer.compare_with_cointegration(mock_coint_results)
    for row in comparison[:10]:
        print(f"  {row['pair']}: r={row['correlation']:.3f}, "
              f"p_coint={row['p_value_coint']}, {row['status']}")
    
    print(analyzer.summary(comparison))