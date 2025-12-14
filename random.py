from datetime import timedelta
import polars as pl

def avg_price(data, time=timedelta(hours=1), price="close", weight="time") -> float:
    """
    Compute an average price weighted by time (TWAP) or volume (VWAP).

    Parameters
    ----------
    data : Dataframe of order book.
    time : Filter out future orders.
    price : Which price to use.
    weight : Which weighting to use.

    Returns average price.
    """
    
    # ensure correct values
    allowed_price = {"open", "close", "high", "low", "average", }
    allowed_weight = {"time", "volume"}

    data = data.filter(pl.col("time_in_hour") <= time)
    
    if price not in allowed_price:
        raise ValueError(f"weight must be one of {allowed_price}")
    if weight not in allowed_weight:
        raise ValueError(f"weight must be one of {allowed_weight}")
    
    if price == "average":
        prices = (data["high"] + data["low"]) / 2
    else:
        prices = data[price]
    
    if weight == "time":
        return prices.mean()
    
    else:
        volume = data["volume"]
        
        return (prices * volume).sum() / volume.sum()
