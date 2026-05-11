import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from config import data_config


class MOEXLoader:
    """Загрузчик данных через MOEX ISS API"""
    
    def __init__(self, use_cache=True):
        self.base_url = "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/tqbr/securities"
        self.use_cache = use_cache
        self.cache_dir = "data/cache"
        
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.last_load_info = {
            "used_cache": False,
            "cache_valid": None,
            "cache_period": {"start": None, "end": None},
        }
    
    def _get_cache_path(self, tickers, start_date, end_date):
        tickers_str = "_".join(tickers)
        return os.path.join(self.cache_dir, f"moex_{tickers_str}_{start_date}_{end_date}.csv")
    
    def load_prices(self, tickers=None, start_date=None, end_date=None, force_refresh=False):
        if tickers is None:
            tickers = data_config.tickers
        if start_date is None:
            start_date = data_config.start_date
        if end_date is None:
            end_date = data_config.end_date
        
        cache_path = self._get_cache_path(tickers, start_date, end_date)

        requested_start = pd.to_datetime(start_date)
        requested_end = pd.to_datetime(end_date)
        self.last_load_info = {
            "used_cache": False,
            "cache_valid": None,
            "cache_period": {"start": None, "end": None},
        }

        if self.use_cache and (not force_refresh) and os.path.exists(cache_path):
            cache_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cache_start = cache_df.index.min() if not cache_df.empty else None
            cache_end = cache_df.index.max() if not cache_df.empty else None
            cache_valid = bool(
                cache_start is not None
                and cache_end is not None
                and cache_start <= requested_start
                and cache_end >= requested_end
            )
            self.last_load_info["cache_period"] = {
                "start": str(cache_start.date()) if cache_start is not None else None,
                "end": str(cache_end.date()) if cache_end is not None else None,
            }
            self.last_load_info["cache_valid"] = cache_valid

            if cache_valid:
                self.last_load_info["used_cache"] = True
                print(f"Загружено из валидного кэша: {cache_path}")
                return cache_df
            print(f"Кэш не покрывает период {start_date}..{end_date}, выполняю свежую загрузку: {cache_path}")
        
        prices = pd.DataFrame()
        
        for ticker in tickers:
            print(f"Загружаю {ticker}...")
            
            url = f"{self.base_url}/{ticker}.json"
            params = {
                "from": start_date,
                "to": end_date,
                "iss.meta": "off",
                "limit": 500,
            }
            
            all_rows = []
            columns = None
            start = 0
            while True:
                paged_params = {**params, "start": start}
                response = requests.get(url, params=paged_params, timeout=30)
                if response.status_code != 200:
                    print(f"  ✗ Ошибка загрузки {ticker}: {response.status_code}")
                    break
                data = response.json()
                history_payload = data.get("history", {})
                batch = history_payload.get("data", [])
                columns = history_payload.get("columns", columns)
                if not batch:
                    break
                all_rows.extend(batch)
                cursor_payload = data.get("history.cursor", {})
                cursor_data = cursor_payload.get("data", [])
                total = cursor_data[0][1] if cursor_data else None
                if total is None or len(all_rows) >= int(total):
                    break
                start += len(batch)

            if not all_rows or columns is None:
                print(f"  ✗ Нет данных по {ticker}")
                continue

            df = pd.DataFrame(all_rows, columns=columns)
            if df.empty:
                continue

            df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
            df.set_index('TRADEDATE', inplace=True)
            df['close'] = df['close'].astype(float)

            prices[ticker] = df['close']
            print(f"  ✓ {len(df)} дней")
        
        if self.use_cache and len(prices) > 0:
            prices.to_csv(cache_path)
            print(f"\nСохранено в кэш: {cache_path}")
        
        return prices


if __name__ == "__main__":
    loader = MOEXLoader(use_cache=True)
    prices = loader.load_prices()
    
    print("\nРезультат:")
    print(prices.head())
    print(f"\nВсего дней: {len(prices)}")
    print(f"Акции: {list(prices.columns)}")
