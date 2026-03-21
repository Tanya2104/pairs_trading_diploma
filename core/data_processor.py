import pandas as pd
import numpy as np
import os
import json
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Класс для обработки и очистки временных рядов цен"""
    
    def __init__(self, prices: pd.DataFrame = None):
        self.raw_prices = prices.copy() if prices is not None else None
        self.processed_prices = None
        self.metadata = {}
        self.processed_dir = "data/processed"
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def check_quality(self) -> Dict:
        """Проверка качества данных"""
        if self.raw_prices is None:
            return {'error': 'Нет данных'}
        
        quality_report = {
            'n_days': len(self.raw_prices),
            'n_tickers': len(self.raw_prices.columns),
            'total_cells': len(self.raw_prices) * len(self.raw_prices.columns),
            'missing_values': self.raw_prices.isnull().sum().sum(),
            'missing_by_ticker': self.raw_prices.isnull().sum().to_dict(),
            'zero_values': (self.raw_prices == 0).sum().sum(),
            'negative_values': (self.raw_prices < 0).sum().sum()
        }
        
        total_cells = quality_report['total_cells']
        if total_cells > 0:
            quality_report['missing_pct'] = quality_report['missing_values'] / total_cells * 100
        else:
            quality_report['missing_pct'] = 0
        
        empty_tickers = []
        for ticker in self.raw_prices.columns:
            if self.raw_prices[ticker].isnull().all():
                empty_tickers.append(ticker)
        quality_report['empty_tickers'] = empty_tickers
        
        self.metadata['quality'] = quality_report
        return quality_report
    
    def remove_empty_tickers(self, threshold: float = 0.5) -> pd.DataFrame:
        """Удаляет тикеры с большим количеством пропусков"""
        data = self.processed_prices if self.processed_prices is not None else self.raw_prices
        
        if data is None:
            logger.warning("Нет данных для обработки")
            return None
        
        missing_pct = data.isnull().sum() / len(data)
        keep_tickers = missing_pct[missing_pct <= threshold].index.tolist()
        
        removed = missing_pct[missing_pct > threshold].index.tolist()
        if removed:
            logger.info(f"Удалены тикеры с >{threshold*100}% пропусков: {removed}")
        
        self.processed_prices = data[keep_tickers].copy()
        self.metadata['removed_tickers'] = removed
        self.metadata['keep_tickers'] = keep_tickers
        
        return self.processed_prices
    
    def synchronize_dates(self) -> pd.DataFrame:
        """Синхронизирует временные ряды по датам"""
        if self.processed_prices is None:
            logger.warning("Нет данных для синхронизации (сначала вызовите remove_empty_tickers)")
            return None
        
        n_before = len(self.processed_prices)
        self.processed_prices = self.processed_prices.dropna()
        n_after = len(self.processed_prices)
        
        if n_before != n_after:
            logger.info(f"Синхронизация: удалено {n_before - n_after} дней с пропусками")
        
        self.metadata['n_days_before'] = n_before
        self.metadata['n_days_after'] = n_after
        
        return self.processed_prices
    
    def save_processed(self, filename: str = "prices_processed.csv") -> None:
        """Сохраняет обработанные данные в файл"""
        if self.processed_prices is None:
            logger.warning("Нет обработанных данных для сохранения")
            return
        filepath = os.path.join(self.processed_dir, filename)
        self.processed_prices.to_csv(filepath)
        logger.info(f"Обработанные данные сохранены в {filepath}")
    
    def load_processed(self, filename: str = "prices_processed.csv") -> pd.DataFrame:
        """Загружает ранее обработанные данные из файла"""
        filepath = os.path.join(self.processed_dir, filename)
        if os.path.exists(filepath):
            self.processed_prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Загружены обработанные данные из {filepath}")
            return self.processed_prices
        else:
            logger.warning(f"Файл {filepath} не найден")
            return None
    
    def save_metadata(self, filename: str = "metadata.json") -> None:
        """Сохраняет метаданные обработки в JSON"""
        filepath = os.path.join(self.processed_dir, filename)
        
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            return obj
        
        metadata_serializable = {}
        for key, value in self.metadata.items():
            if isinstance(value, dict):
                metadata_serializable[key] = {k: convert(v) for k, v in value.items()}
            else:
                metadata_serializable[key] = convert(value)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata_serializable, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Метаданные сохранены в {filepath}")
    
    def get_processed_data(self) -> pd.DataFrame:
        """Возвращает обработанные данные"""
        return self.processed_prices
    
    def get_summary(self) -> str:
        """Возвращает текстовую сводку"""
        if 'quality' not in self.metadata:
            self.check_quality()
        
        data_for_summary = self.processed_prices if self.processed_prices is not None else self.raw_prices
        
        if data_for_summary is None:
            return "❌ Нет данных для анализа"
        
        return f"""
        📊 Сводка по данным
        {'='*50}
        Период: {data_for_summary.index[0].date()} - {data_for_summary.index[-1].date()}
        Всего дней: {len(data_for_summary)}
        Всего тикеров: {len(data_for_summary.columns)}
        
        Качество данных:
          Пропуски: {self.metadata['quality']['missing_values']} из {self.metadata['quality']['total_cells']} ({self.metadata['quality']['missing_pct']:.2f}%)
        
        После обработки:
          Осталось дней: {len(self.processed_prices) if self.processed_prices is not None else 'не обработано'}
          Осталось тикеров: {len(self.processed_prices.columns) if self.processed_prices is not None else 'не обработано'}
        """


# ============= ТЕСТ =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from core.data_loader import MOEXLoader
    from config import data_config
    
    # 1. Загружаем сырые данные
    loader = MOEXLoader(use_cache=True)
    prices = loader.load_prices(
        tickers=data_config.tickers,
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print(f"Загружено: {prices.shape}")
    
    # 2. Обрабатываем
    processor = DataProcessor(prices)
    quality = processor.check_quality()
    print(processor.get_summary())
    
    # 3. Чистим данные
    processor.remove_empty_tickers(threshold=0.3)
    processor.synchronize_dates()
    
    # 4. Сохраняем обработанные данные
    processor.save_processed("prices_2023_5tickers.csv")
    processor.save_metadata("metadata_2023_5tickers.json")
    
    # 5. Проверяем, что загружается обратно
    print("\n" + "="*50)
    print("Проверка загрузки сохранённых данных:")
    processor2 = DataProcessor()
    loaded = processor2.load_processed("prices_2023_5tickers.csv")
    
    if loaded is not None:
        print(loaded.head())
        print(f"\n✅ Загружено: {loaded.shape}")
    else:
        print("❌ Не удалось загрузить данные")