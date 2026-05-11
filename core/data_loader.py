import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
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
            "tickers_loaded": [],
            "failed_tickers": [],
            "loaded_period": {"start": None, "end": None},
        }
    
    def _get_cache_path(self, tickers, start_date, end_date):
        tickers_str = "_".join(tickers)
        return os.path.join(self.cache_dir, f"moex_{tickers_str}_{start_date}_{end_date}.csv")
    
    def load_prices(self, tickers=None, start_date=None, end_date=None, force_refresh=False, progress_callback=None, timeout=20, max_pages=200, request_pause=0.15):
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
            "tickers_loaded": [],
            "failed_tickers": [],
            "loaded_period": {"start": None, "end": None},
        }

        if self.use_cache and (not force_refresh) and os.path.exists(cache_path):
            cache_df = pd.read_csv(cache_path, index_col=0, parse_dates=True).sort_index()
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
                self.last_load_info["loaded_period"] = {
                    "start": str(cache_start.date()) if cache_start is not None else None,
                    "end": str(cache_end.date()) if cache_end is not None else None,
                }
                filtered_cache = cache_df.loc[(cache_df.index >= requested_start) & (cache_df.index <= requested_end)]
                print(f"Загружено из валидного кэша: {cache_path}")
                return filtered_cache
            print(f"Кэш не покрывает период {start_date}..{end_date}, выполняю свежую загрузку: {cache_path}")
        
        prices = pd.DataFrame()
        loaded_tickers = []
        failed_tickers = []

        for t_idx, ticker in enumerate(tickers, start=1):
            try:
                print(f"Загружаю {ticker} ({t_idx}/{len(tickers)})...")
                
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
                prev_start = -1
                for page in range(1, max_pages + 1):
                    if start <= prev_start:
                        print(f"  ⚠ Остановка {ticker}: offset не увеличивается (start={start}, prev={prev_start})")
                        break
                    paged_params = {**params, "start": start}
                    try:
                        response = requests.get(url, params=paged_params, timeout=timeout)
                        response.raise_for_status()
                    except requests.RequestException as exc:
                        print(f"  ✗ Ошибка загрузки {ticker} (страница {page}, offset={start}): {exc}")
                        failed_tickers.append(ticker)
                        break

                    data = response.json()
                    history_payload = data.get("history", {})
                    batch = history_payload.get("data", [])
                    columns = history_payload.get("columns", columns)
                    batch_rows = len(batch)
                    print(f"  ↳ {ticker}: page={page}, offset={start}, rows={batch_rows}, total_rows={len(all_rows)+batch_rows}")
                    if progress_callback is not None:
                        progress_callback(ticker=ticker, ticker_index=t_idx, total_tickers=len(tickers), page=page, offset=start, batch_rows=batch_rows, total_rows=len(all_rows)+batch_rows)

                    if batch_rows == 0:
                        break

                    all_rows.extend(batch)
                    cursor_payload = data.get("history.cursor", {})
                    cursor_data = cursor_payload.get("data", [])
                    total = cursor_data[0][1] if cursor_data else None

                    prev_start = start
                    start += batch_rows

                    if total is not None and len(all_rows) >= int(total):
                        break
                    time.sleep(request_pause)
                else:
                    print(f"  ⚠ Достигнут лимит страниц max_pages={max_pages} для {ticker}")

                if not all_rows or columns is None:
                    print(f"  ⚠ Нет данных по {ticker} за период {start_date}..{end_date}")
                    failed_tickers.append(ticker)
                    continue

                df = pd.DataFrame(all_rows, columns=columns)
                if df.empty:
                    print(f"  ⚠ Пустой ответ MOEX по {ticker} за период {start_date}..{end_date}")
                    failed_tickers.append(ticker)
                    continue

                # Нормализуем названия колонок MOEX: нижний регистр + обрезка пробелов.
                df.columns = [str(col).strip().lower() for col in df.columns]
                available_columns = list(df.columns)

                # Диагностика загруженных данных по тикеру.
                print(
                    f"  ℹ Диагностика {ticker}: строк={len(df)}, "
                    f"колонки={available_columns}"
                )

                if "tradedate" not in df.columns:
                    raise ValueError(
                        f"Не найдена колонка tradedate в данных MOEX для {ticker}. "
                        f"Доступные колонки: {available_columns}"
                    )

                close_candidates = [
                    "close",
                    "legalcloseprice",
                    "waprice",
                    "marketprice2",
                    "admittedquote",
                ]
                selected_close_column = next(
                    (candidate for candidate in close_candidates if candidate in df.columns),
                    None,
                )

                if selected_close_column is None:
                    raise ValueError(
                        "Не найдена колонка цены закрытия в данных MOEX. "
                        f"Тикер: {ticker}. Доступные колонки: {available_columns}"
                    )

                print(f"  ℹ Выбрана колонка close для {ticker}: {selected_close_column}")

                df["tradedate"] = pd.to_datetime(df["tradedate"])
                df.set_index("tradedate", inplace=True)
                df["close"] = pd.to_numeric(df[selected_close_column], errors="coerce")
                df = df.dropna(subset=["close"])
                if df.empty:
                    print(f"  ⚠ После преобразования цен нет валидных данных по {ticker}")
                    failed_tickers.append(ticker)
                    continue

                prices[ticker] = df["close"]
                loaded_tickers.append(ticker)
                print(f"  ✓ {len(df)} дней")
            except Exception as exc:
                print(f"  ✗ Критическая ошибка обработки тикера {ticker}: {exc}")
                failed_tickers.append(ticker)
                continue
        
        self.last_load_info["tickers_loaded"] = loaded_tickers
        self.last_load_info["failed_tickers"] = sorted(set(failed_tickers))

        prices = prices.sort_index()
        prices = prices.loc[(prices.index >= requested_start) & (prices.index <= requested_end)]

        loaded_start = prices.index.min() if not prices.empty else None
        loaded_end = prices.index.max() if not prices.empty else None
        self.last_load_info["loaded_period"] = {
            "start": str(loaded_start.date()) if loaded_start is not None else None,
            "end": str(loaded_end.date()) if loaded_end is not None else None,
        }

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
