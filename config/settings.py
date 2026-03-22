from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    """Параметры загрузки данных"""
    tickers: List[str] = None
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    
    def __post_init__(self):
        if self.tickers is None:
            self.tickers = [
                "SBER", "GAZP", "LKOH", "ROSN", "NVTK",
                "TATN", "GMKN", "PLZL", "CHMF", "NLMK",
                "MAGN", "YNDX", "MTSS", "AFLT", "VTBR"
            ]


@dataclass
class StrategyConfig:
    """Параметры торговой стратегии"""
    zscore_window: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.5


@dataclass
class CointegrationConfig:
    """Параметры теста коинтеграции"""
    p_value_threshold: float = 0.05
    max_lags: int = 10


# Глобальные экземпляры
data_config = DataConfig()
strategy_config = StrategyConfig()
coint_config = CointegrationConfig()
