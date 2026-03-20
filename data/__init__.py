from data.data_loader import MOEXLoader

def get_data_loader(source: str = "moex", **kwargs):
    if source == "moex":
        return MOEXLoader(**kwargs)
    elif source == "yfinance":
        from data.yfinance_loader import YFinanceLoader
        return YFinanceLoader(**kwargs)
    elif source == "tinvest":
        from data.tinvest_loader import TInvestLoader
        return TInvestLoader(**kwargs)
    else:
        raise ValueError(f"Неизвестный источник: {source}")