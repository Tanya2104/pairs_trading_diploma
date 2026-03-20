import requests
import pandas as pd
import os

class MOEXLoader:
    """Загрузчик данных через MOEX ISS API"""
    
    def __init__(self, use_cache=True):
        self.base_url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/tqbr/securities"
        self.use_cache = use_cache
        self.cache_dir = "data/cache"
        
        # Создаём папку для кэша, если её нет
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, tickers, start_date, end_date):
        """Формирует имя файла для кэша"""
        tickers_str = "_".join(tickers)
        return os.path.join(self.cache_dir, f"moex_{tickers_str}_{start_date}_{end_date}.csv")
    
    def load_prices(self, tickers, start_date, end_date):
        """
        Загружает цены закрытия для списка тикеров
        """
        # Проверяем кэш
        cache_path = self._get_cache_path(tickers, start_date, end_date)
        
        if self.use_cache and os.path.exists(cache_path):
            print(f"Загружено из кэша: {cache_path}")
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        
        prices = pd.DataFrame()
        
        for ticker in tickers:
            print(f"Загружаю {ticker}...")
            
            url = f"{self.base_url}/{ticker}/candles.json"
            params = {
                "from": start_date,
                "to": end_date,
                "interval": 24
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            columns = data['candles']['columns']
            rows = data['candles']['data']
            
            df = pd.DataFrame(rows, columns=columns)
            df['begin'] = pd.to_datetime(df['begin'])
            df.set_index('begin', inplace=True)
            df['close'] = df['close'].astype(float)
            
            prices[ticker] = df['close']
            print(f"  ✓ {len(df)} дней")
        
        # Убираем дни с пропусками
        prices = prices.dropna()
        
        # Сохраняем в кэш
        if self.use_cache and len(prices) > 0:
            prices.to_csv(cache_path)
            print(f"\nСохранено в кэш: {cache_path}")
        
        return prices


# Тест
if __name__ == "__main__":
    loader = MOEXLoader(use_cache=True)
    tickers = ["SBER", "GAZP"]
    prices = loader.load_prices(tickers, "2023-01-01", "2023-12-31")
    
    print("\nРезультат:")
    print(prices.head())
    print(f"\nВсего дней: {len(prices)}")
    print(f"Акции: {list(prices.columns)}")