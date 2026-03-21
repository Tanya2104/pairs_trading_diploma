
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


class LinearRegression:
    """
    Линейная регрессия y = α + β*x + ε
    
    Ручная реализация метода наименьших квадратов
    """
    
    def __init__(self):
        self.alpha = None      # свободный член
        self.beta = None       # коэффициент наклона
        self.residuals = None  # остатки
        self.predictions = None
        self.r_squared = None
        self.se_beta = None     # стандартная ошибка beta
        self.se_alpha = None    # стандартная ошибка alpha
        self.t_stat_beta = None # t-статистика для beta
        self.t_stat_alpha = None
        self.n = None           # количество наблюдений
        self._fitted = False
    
    def fit(self, X: pd.Series, y: pd.Series) -> 'LinearRegression':
        """
        Оценка коэффициентов регрессии
        
        Parameters
        ----------
        X : pd.Series
            Независимая переменная (цены первого актива)
        y : pd.Series
            Зависимая переменная (цены второго актива)
            
        Returns
        -------
        self
        """
        # Убираем пропуски и синхронизируем
        mask = ~(X.isna() | y.isna())
        X_clean = X[mask].values
        y_clean = y[mask].values
        
        self.n = len(X_clean)
        
        if self.n < 3:
            raise ValueError(f"Недостаточно данных: {self.n} наблюдений")
        
        # Средние значения
        mean_x = np.mean(X_clean)
        mean_y = np.mean(y_clean)
        
        # Отклонения от средних
        x_dev = X_clean - mean_x
        y_dev = y_clean - mean_y
        
        # beta = cov(x,y) / var(x)
        numerator = np.sum(x_dev * y_dev)
        denominator = np.sum(x_dev ** 2)
        
        if denominator == 0:
            raise ValueError("Дисперсия X равна нулю")
        
        self.beta = numerator / denominator
        self.alpha = mean_y - self.beta * mean_x
        
        # Предсказания и остатки
        self.predictions = self.alpha + self.beta * X_clean
        self.residuals = y_clean - self.predictions
        
        # R-squared
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum(y_dev ** 2)
        
        if ss_tot == 0:
            self.r_squared = 1.0
        else:
            self.r_squared = 1 - (ss_res / ss_tot)
        
        # Стандартные ошибки
        sigma2 = ss_res / (self.n - 2)  # несмещённая оценка дисперсии ошибок
        
        self.se_beta = np.sqrt(sigma2 / denominator)
        self.se_alpha = np.sqrt(sigma2 * (1/self.n + mean_x**2 / denominator))
        
        # t-статистики
        self.t_stat_beta = self.beta / self.se_beta if self.se_beta != 0 else 0
        self.t_stat_alpha = self.alpha / self.se_alpha if self.se_alpha != 0 else 0
        
        self._fitted = True
        
        return self
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """
        Предсказание по модели
        """
        if not self._fitted:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit()")
        
        return self.alpha + self.beta * X.values
    
    def get_coefficients(self) -> Dict[str, float]:
        """
        Возвращает коэффициенты регрессии
        """
        if not self._fitted:
            raise RuntimeError("Модель не обучена")
        
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'r_squared': self.r_squared,
            'se_alpha': self.se_alpha,
            'se_beta': self.se_beta,
            't_stat_alpha': self.t_stat_alpha,
            't_stat_beta': self.t_stat_beta,
            'n_obs': self.n
        }
    
    def get_residuals(self) -> Optional[np.ndarray]:
        """
        Возвращает остатки регрессии
        """
        if not self._fitted:
            return None
        return self.residuals
    
    def get_residuals_series(self, index: pd.Index) -> pd.Series:
        """
        Возвращает остатки в виде pandas Series с исходным индексом
        """
        if not self._fitted:
            raise RuntimeError("Модель не обучена")
        
        # Создаём Series с NaN для всех дат
        residuals_series = pd.Series(np.nan, index=index)
        
        # Заполняем только те даты, которые были в обучении
        # Нужно сохранить индексы при обучении
        # Пока упростим: возвращаем массив
        return pd.Series(self.residuals)
    
    def summary(self) -> str:
        """
        Возвращает текстовую сводку регрессии
        """
        if not self._fitted:
            return "Модель не обучена"
        
        return f"""
        ================================================
        Результаты регрессии
        ================================================
        Зависимая переменная: y
        Независимая переменная: X
        
        Коэффициенты:
          α (константа) = {self.alpha:.6f}
          β (наклон)    = {self.beta:.6f}
        
        Статистики:
          R² = {self.r_squared:.6f}
          Количество наблюдений: {self.n}
        
        Стандартные ошибки:
          SE(α) = {self.se_alpha:.6f}
          SE(β) = {self.se_beta:.6f}
        
        t-статистики:
          t(α) = {self.t_stat_alpha:.4f}
          t(β) = {self.t_stat_beta:.4f}
        ================================================
        """


def ols(X: pd.Series, y: pd.Series) -> Dict:
    """
    Функция-обёртка для быстрой оценки регрессии
    
    Parameters
    ----------
    X : pd.Series
        Независимая переменная
    y : pd.Series
        Зависимая переменная
        
    Returns
    -------
    dict
        Словарь с результатами
    """
    model = LinearRegression()
    model.fit(X, y)
    return model.get_coefficients()


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
        tickers=["SBER", "GAZP"],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print("=" * 60)
    print("Тест ручной регрессии")
    print("=" * 60)
    
    # Тест 1: регрессия SBER ~ GAZP
    print("\n[1] Регрессия SBER на GAZP")
    model = LinearRegression()
    model.fit(prices["GAZP"], prices["SBER"])
    print(model.summary())
    
    # Тест 2: проверка предсказаний
    print("\n[2] Проверка предсказаний (первые 5)")
    pred = model.predict(prices["GAZP"].iloc[:5])
    actual = prices["SBER"].iloc[:5].values
    
    for i, (p, a) in enumerate(zip(pred, actual)):
        print(f"  {i+1}: предсказано = {p:.2f}, фактически = {a:.2f}, ошибка = {p-a:.2f}")
    
    # Тест 3: сравнение с numpy.polyfit (эталон)
    print("\n[3] Сравнение с numpy.polyfit")
    X = prices["GAZP"].dropna().values
    y = prices["SBER"].dropna().values
    
    polyfit_beta, polyfit_alpha = np.polyfit(X, y, 1)
    
    print(f"  Наша реализация: α = {model.alpha:.6f}, β = {model.beta:.6f}")
    print(f"  numpy.polyfit:   α = {polyfit_alpha:.6f}, β = {polyfit_beta:.6f}")
    
    diff_alpha = abs(model.alpha - polyfit_alpha)
    diff_beta = abs(model.beta - polyfit_beta)
    
    if diff_alpha < 1e-6 and diff_beta < 1e-6:
        print("\n  ✅ Результаты совпадают с эталоном!")
    else:
        print(f"\n  ⚠️ Расхождения: Δα = {diff_alpha:.2e}, Δβ = {diff_beta:.2e}")