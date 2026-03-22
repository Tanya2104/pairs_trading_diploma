"""
Р СџР С•Р С‘РЎРѓР С” Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р Р…РЎвЂ№РЎвЂ¦ Р С—Р В°РЎР‚ Р СР ВµРЎвЂљР С•Р Т‘Р С•Р С Р В­Р Р…Р С–Р В»Р В°-Р вЂњРЎР‚РЎРЊР Р…Р Т‘Р В¶Р ВµРЎР‚Р В°
Р РЋРЎР‚Р В°Р Р†Р Р…Р ВµР Р…Р С‘Р Вµ РЎРѓ Р С”Р С•РЎР‚РЎР‚Р ВµР В»РЎРЏРЎвЂ Р С‘Р С•Р Р…Р Р…РЎвЂ№Р С Р В°Р Р…Р В°Р В»Р С‘Р В·Р С•Р С
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
    Р СџР С•Р С‘РЎРѓР С” Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р Р…РЎвЂ№РЎвЂ¦ Р С—Р В°РЎР‚ Р СР ВµРЎвЂљР С•Р Т‘Р С•Р С Р В­Р Р…Р С–Р В»Р В°-Р вЂњРЎР‚РЎРЊР Р…Р Т‘Р В¶Р ВµРЎР‚Р В°
    """
    
    def __init__(self, prices: pd.DataFrame, p_value_threshold: float = 0.05):
        """
        Parameters
        ----------
        prices : pd.DataFrame
            Р В¦Р ВµР Р…РЎвЂ№ Р В·Р В°Р С”РЎР‚РЎвЂ№РЎвЂљР С‘РЎРЏ (Р С‘Р Р…Р Т‘Р ВµР С”РЎРѓ - Р Т‘Р В°РЎвЂљР В°, Р С”Р С•Р В»Р С•Р Р…Р С”Р С‘ - РЎвЂљР С‘Р С”Р ВµРЎР‚РЎвЂ№)
        p_value_threshold : float
            Р СџР С•РЎР‚Р С•Р С– p-value Р Т‘Р В»РЎРЏ Р С•РЎвЂљР В±Р С•РЎР‚Р В° Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р Р…РЎвЂ№РЎвЂ¦ Р С—Р В°РЎР‚
        """
        self.prices = prices
        self.threshold = p_value_threshold
        self.results = []
        self.correlation_analyzer = CorrelationAnalyzer(prices)
        self.adf_tester = ADFTest(max_lags=10, autolag='AIC')
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Р В Р В°РЎРѓРЎвЂЎРЎвЂРЎвЂљ Р С—Р ВµРЎР‚Р С‘Р С•Р Т‘Р В° Р С—Р С•Р В»РЎС“РЎР‚Р В°РЎРѓР С—Р В°Р Т‘Р В° РЎРѓР С—РЎР‚Р ВµР Т‘Р В°
        Half-life = -ln(2) / РћС‘, Р С–Р Т‘Р Вµ РћС‘ РІР‚вЂќ Р С”Р С•РЎРЊРЎвЂћРЎвЂћР С‘РЎвЂ Р С‘Р ВµР Р…РЎвЂљ Р В°Р Р†РЎвЂљР С•РЎР‚Р ВµР С–РЎР‚Р ВµРЎРѓРЎРѓР С‘Р С‘
        """
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        # Р РЋР С‘Р Р…РЎвЂ¦РЎР‚Р С•Р Р…Р С‘Р В·Р С‘РЎР‚РЎС“Р ВµР С
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
        except (ValueError, TypeError, ZeroDivisionError, ImportError):
            half_life = np.inf
        
        return half_life
    
    def test_pair(self, ticker1: str, ticker2: str) -> Optional[Dict]:
        """
        Р СџРЎР‚Р С•Р Р†Р ВµРЎР‚Р С”Р В° Р С•Р Т‘Р Р…Р С•Р в„– Р С—Р В°РЎР‚РЎвЂ№ Р Р…Р В° Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р В°РЎвЂ Р С‘РЎР‹
        
        Returns
        -------
        dict or None
            Р В Р ВµР В·РЎС“Р В»РЎРЉРЎвЂљР В°РЎвЂљРЎвЂ№ Р В°Р Р…Р В°Р В»Р С‘Р В·Р В° Р С—Р В°РЎР‚РЎвЂ№
        """
        x = self.prices[ticker1]
        y = self.prices[ticker2]
        
        # Р Р€Р В±Р С‘РЎР‚Р В°Р ВµР С Р С—РЎР‚Р С•Р С—РЎС“РЎРѓР С”Р С‘
        df_pair = pd.concat([x, y], axis=1).dropna()
        if len(df_pair) < 30:
            return None
        
        X = df_pair.iloc[:, 0]
        Y = df_pair.iloc[:, 1]
        
        # 1. Р В Р ВµР С–РЎР‚Р ВµРЎРѓРЎРѓР С‘РЎРЏ Y ~ X
        model = LinearRegression()
        model.fit(X, Y)
        coeffs = model.get_coefficients()
        residuals = model.get_residuals()
        
        # 2. ADF-РЎвЂљР ВµРЎРѓРЎвЂљ Р С•РЎРѓРЎвЂљР В°РЎвЂљР С”Р С•Р Р†
        adf_result = self.adf_tester.run(pd.Series(residuals))
        
        # 3. Р СџР ВµРЎР‚Р С‘Р С•Р Т‘ Р С—Р С•Р В»РЎС“РЎР‚Р В°РЎРѓР С—Р В°Р Т‘Р В°
        spread = pd.Series(residuals, index=df_pair.index)
        half_life = self._calculate_half_life(spread)
        
        # 4. Р С™Р С•РЎР‚РЎР‚Р ВµР В»РЎРЏРЎвЂ Р С‘РЎРЏ
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
            'is_cointegrated': adf_result['is_stationary'] and adf_result['p_value'] < self.threshold
        }
        
        return result
    
    def find_pairs(self) -> List[Dict]:
        """
        Р СџР ВµРЎР‚Р ВµР В±Р С‘РЎР‚Р В°Р ВµРЎвЂљ Р Р†РЎРѓР Вµ Р С—Р В°РЎР‚РЎвЂ№ Р С‘ Р Р…Р В°РЎвЂ¦Р С•Р Т‘Р С‘РЎвЂљ Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р Р…РЎвЂ№Р Вµ
        """
        tickers = self.prices.columns.tolist()
        self.results = []
        
        total_pairs = len(list(combinations(tickers, 2)))
        print(f"Р С’Р Р…Р В°Р В»Р С‘Р В· {len(tickers)} Р В°Р С”РЎвЂ Р С‘Р в„–, Р Р†РЎРѓР ВµР С–Р С• Р С—Р В°РЎР‚: {total_pairs}")
        
        for t1, t2 in combinations(tickers, 2):
            result = self.test_pair(t1, t2)
            if result:
                if result['is_cointegrated']:
                    print(f"  РІСљвЂ¦ {t1}-{t2}: p={result['p_value']:.4f}, "
                          f"rР’Р†={result['r_squared']:.3f}, r={result['correlation']:.3f}")
                self.results.append(result)
        
        # Р РЋР С•РЎР‚РЎвЂљР С‘РЎР‚Р С•Р Р†Р С”Р В° Р С—Р С• p-value (Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р Р…РЎвЂ№Р Вµ Р Р† Р Р…Р В°РЎвЂЎР В°Р В»Р Вµ)
        self.results.sort(key=lambda x: x['p_value'])
        
        return self.results
    
    def get_best_pair(self) -> Optional[Dict]:
        """Р вЂ™Р С•Р В·Р Р†РЎР‚Р В°РЎвЂ°Р В°Р ВµРЎвЂљ Р В»РЎС“РЎвЂЎРЎв‚¬РЎС“РЎР‹ Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р Р…РЎС“РЎР‹ Р С—Р В°РЎР‚РЎС“"""
        if not self.results:
            self.find_pairs()
        
        cointegrated = [r for r in self.results if r['is_cointegrated']]
        if cointegrated:
            return cointegrated[0]
        return None
    
    def get_comparison_table(self) -> List[Dict]:
        """
        Р вЂ™Р С•Р В·Р Р†РЎР‚Р В°РЎвЂ°Р В°Р ВµРЎвЂљ РЎвЂљР В°Р В±Р В»Р С‘РЎвЂ РЎС“ РЎРѓРЎР‚Р В°Р Р†Р Р…Р ВµР Р…Р С‘РЎРЏ Р С”Р С•РЎР‚РЎР‚Р ВµР В»РЎРЏРЎвЂ Р С‘Р С‘ Р С‘ Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р В°РЎвЂ Р С‘Р С‘
        """
        if not self.results:
            self.find_pairs()
        
        return self.correlation_analyzer.compare_with_cointegration(self.results)


# ============= Р СћР вЂўР РЋР Сћ =============
if __name__ == "__main__":
    from core.data_loader import MOEXLoader
    from core.data_processor import DataProcessor
    from config import data_config
    
    print("=" * 60)
    print("Р СџР С•Р С‘РЎРѓР С” Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р Р…РЎвЂ№РЎвЂ¦ Р С—Р В°РЎР‚ (Р СР ВµРЎвЂљР С•Р Т‘ Р В­Р Р…Р С–Р В»Р В°-Р вЂњРЎР‚РЎРЊР Р…Р Т‘Р В¶Р ВµРЎР‚Р В°)")
    print("=" * 60)
    
    # 1. Р вЂ”Р В°Р С–РЎР‚РЎС“Р В¶Р В°Р ВµР С РЎРѓРЎвЂ№РЎР‚РЎвЂ№Р Вµ Р Т‘Р В°Р Р…Р Р…РЎвЂ№Р Вµ
    loader = MOEXLoader(use_cache=True)
    raw_prices = loader.load_prices(
        tickers=data_config.tickers[:8],  # Р С—Р ВµРЎР‚Р Р†РЎвЂ№Р Вµ 8 Р Т‘Р В»РЎРЏ РЎвЂљР ВµРЎРѓРЎвЂљР В°
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print(f"\nР вЂ”Р В°Р С–РЎР‚РЎС“Р В¶Р ВµР Р…Р С• РЎРѓРЎвЂ№РЎР‚РЎвЂ№РЎвЂ¦ Р Т‘Р В°Р Р…Р Р…РЎвЂ№РЎвЂ¦: {raw_prices.shape}")
    
    # 2. Р С›Р В±РЎР‚Р В°Р В±Р В°РЎвЂљРЎвЂ№Р Р†Р В°Р ВµР С Р Т‘Р В°Р Р…Р Р…РЎвЂ№Р Вµ
    print("\nР С›Р В±РЎР‚Р В°Р В±Р С•РЎвЂљР С”Р В° Р Т‘Р В°Р Р…Р Р…РЎвЂ№РЎвЂ¦...")
    processor = DataProcessor(raw_prices)
    processor.check_quality()
    processor.remove_empty_tickers(threshold=0.3)
    processor.synchronize_dates()
    
    prices = processor.get_processed_data()
    print(f"Р СџР С•РЎРѓР В»Р Вµ Р С•Р В±РЎР‚Р В°Р В±Р С•РЎвЂљР С”Р С‘: {prices.shape}")
    
    # 3. Р СџР С•Р С‘РЎРѓР С” Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р Р…РЎвЂ№РЎвЂ¦ Р С—Р В°РЎР‚
    tester = CointegrationTester(prices, p_value_threshold=0.05)
    results = tester.find_pairs()
    
    # 4. Р вЂ™РЎвЂ№Р Р†Р С•Р Т‘ РЎР‚Р ВµР В·РЎС“Р В»РЎРЉРЎвЂљР В°РЎвЂљР С•Р Р†
    print(f"\n{'='*60}")
    print("Р В Р ВµР В·РЎС“Р В»РЎРЉРЎвЂљР В°РЎвЂљРЎвЂ№ Р С—Р С•Р С‘РЎРѓР С”Р В° Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р Р…РЎвЂ№РЎвЂ¦ Р С—Р В°РЎР‚")
    print(f"{'='*60}")
    
    cointegrated = [r for r in results if r['is_cointegrated']]
    print(f"\nР СњР В°Р в„–Р Т‘Р ВµР Р…Р С• Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р С‘РЎР‚Р С•Р Р†Р В°Р Р…Р Р…РЎвЂ№РЎвЂ¦ Р С—Р В°РЎР‚: {len(cointegrated)}")
    
    for r in cointegrated[:10]:
        print(f"\n  {r['pair'][0]} - {r['pair'][1]}:")
        print(f"    p-value: {r['p_value']:.6f}")
        print(f"    РћР†: {r['beta']:.4f}")
        print(f"    RР’Р†: {r['r_squared']:.4f}")
        print(f"    Р С™Р С•РЎР‚РЎР‚Р ВµР В»РЎРЏРЎвЂ Р С‘РЎРЏ: {r['correlation']:.4f}")
        print(f"    Half-life: {r['half_life']:.1f} Р Т‘Р Р…Р ВµР в„–")
    
    # 5. Р вЂєРЎС“РЎвЂЎРЎв‚¬Р В°РЎРЏ Р С—Р В°РЎР‚Р В°
    best = tester.get_best_pair()
    if best:
        print(f"\n{'='*60}")
        print(f"Р вЂєРЎС“РЎвЂЎРЎв‚¬Р В°РЎРЏ Р С—Р В°РЎР‚Р В°: {best['pair'][0]} - {best['pair'][1]}")
        print(f"{'='*60}")
        print(f"  p-value: {best['p_value']:.6f}")
        print(f"  РћР†: {best['beta']:.4f}")
        print(f"  РћВ±: {best['alpha']:.2f}")
        print(f"  RР’Р†: {best['r_squared']:.4f}")
        print(f"  Р С™Р С•РЎР‚РЎР‚Р ВµР В»РЎРЏРЎвЂ Р С‘РЎРЏ: {best['correlation']:.4f}")
        print(f"  Half-life: {best['half_life']:.1f} Р Т‘Р Р…Р ВµР в„–")
    
    # 6. Р РЋРЎР‚Р В°Р Р†Р Р…Р С‘РЎвЂљР ВµР В»РЎРЉР Р…Р В°РЎРЏ РЎвЂљР В°Р В±Р В»Р С‘РЎвЂ Р В°
    print(f"\n{'='*60}")
    print("Р РЋРЎР‚Р В°Р Р†Р Р…Р ВµР Р…Р С‘Р Вµ Р С”Р С•РЎР‚РЎР‚Р ВµР В»РЎРЏРЎвЂ Р С‘Р С‘ Р С‘ Р С”Р С•Р С‘Р Р…РЎвЂљР ВµР С–РЎР‚Р В°РЎвЂ Р С‘Р С‘")
    print(f"{'='*60}")
    
    comparison = tester.get_comparison_table()
    for row in comparison[:15]:
        p_coint = f"{row['p_value_coint']:.6f}" if row['p_value_coint'] else "None"
        print(f"  {row['pair']}: r={row['correlation']:.3f}, p_coint={p_coint}, {row['status']}")
    
    print(tester.correlation_analyzer.summary(comparison))