"""
data_handler.py
Multi-timeframe data preparation module for the VectorBT validation engine.
Refactored to use DuckDB for high-performance CSV loading and filtering.
"""

import duckdb
import pandas as pd
import numpy as np
import gc
import os
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta

def load_kraken_data_windowed(filepath: str, lookback_years: int) -> pd.DataFrame:
    """
    Load Kraken OHLCV data from CSV file using chunking to save memory.
    Only loads data within the lookback window.
    
    Args:
        filepath: Path to the Kraken CSV file
        lookback_years: Number of years of data to load from the end of file
        
    Returns:
        DataFrame with datetime index and OHLCV columns
    """
    # Initialize DuckDB connection
    con = duckdb.connect(database=':memory:')

    # 1. Get the last timestamp efficiently
    # Kraken CSVs are headerless: Timestamp, Open, High, Low, Close, Volume, Trades
    # we treat them as column0..column6
    try:
        # We use a limit query or aggregate max to find the end
        query_max = f"SELECT max(column0) FROM read_csv('{filepath}', header=False, auto_detect=True)"
        last_ts = con.execute(query_max).fetchone()[0]

        if last_ts is None:
            raise ValueError(f"Could not determine valid timetamp from {filepath}")

    except Exception as e:
        raise ValueError(f"Error reading file metadata: {e}")

    # Calculate cutoff
    seconds_per_year = 31536000
    cutoff_ts = last_ts - (lookback_years * seconds_per_year)

    dt_start = datetime.utcfromtimestamp(cutoff_ts)
    dt_end = datetime.utcfromtimestamp(last_ts)
    print(f"Filtering data: Loading from {dt_start} to {dt_end}")

    # 2. Load and Filter Data using SQL
    # We perform projection (selecting columns), filtering (WHERE), 
    # and type conversion (to_timestamp) inside the DB engine.
    query_load = f"""
        SELECT 
            -- Convert Unix Epoch (seconds) to Timestamp directly
            to_timestamp(column0) as Timestamp,
            column1::DOUBLE as Open,
            column2::DOUBLE as High,
            column3::DOUBLE as Low,
            column4::DOUBLE as Close,
            column5::DOUBLE as Volume
        FROM read_csv('{filepath}', header=False, auto_detect=True)
        WHERE column0 >= {cutoff_ts}
        ORDER BY column0 ASC
    """

    # Execute and fetch as Pandas DataFrame
    # DuckDB uses Apache Arrow under the hood for zero-copy data transfer where possible
    df = con.execute(query_load).df()
    
    # Clean up DB connection
    con.close()

    if df.empty:
        raise ValueError("No data found within the specified lookback period.")

    # 3. Final Formatting
    df.set_index('Timestamp', inplace=True)
    
    # Ensure UTC timezone awareness (DuckDB returns naive timestamps by default usually)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    
    return df

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV data to a higher timeframe.
    """
    # Convert common timeframe notation to pandas 2.x compatible format
    tf_conversion = {
        '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
        '1d': '1D', '1D': '1D', '1w': '1W', '1W': '1W',
    }
    
    pandas_tf = tf_conversion.get(timeframe, timeframe)
    
    ohlc_agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    resampled = df.resample(pandas_tf).agg(ohlc_agg)
    resampled.dropna(inplace=True)
    
    return resampled

def prepare_data(
    symbol: str,
    data_dir: str = "data/kraken",
    timeframes: Optional[list] = None,
    lookback_years: int = 2,
) -> Dict[str, pd.DataFrame]:
    """
    Prepare multi-timeframe data from a single 1-minute source file.
    Only loads the requested lookback window into memory.
    """
    if timeframes is None:
        timeframes = ['15m', '30m', '1h', '4h', '1d']
    
    # Construct filepath
    filepath = Path(data_dir) / f"{symbol}_1.csv"
    
    # Check if file exists (with fallback options)
    if not filepath.exists():
        alt_paths = [
            Path(data_dir) / f"{symbol}-1m.csv",
            Path(data_dir) / f"{symbol}.csv",
            Path(data_dir) / f"{symbol}_1m.csv",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                filepath = alt_path
                break
        else:
            raise FileNotFoundError(
                f"Could not find data file for {symbol} in {data_dir}."
            )
    
    # Load windowed 1-minute data directly
    print(f"Loading 1-minute data from {filepath} (Last {lookback_years} years)...")
    
    # Uses the new memory-efficient loader
    df_1m = load_kraken_data_windowed(str(filepath), lookback_years)
    
    print(f"Loaded {len(df_1m):,} bars from {df_1m.index[0]} to {df_1m.index[-1]}")
    print(f"Total time duration: {(df_1m.index[-1] - df_1m.index[0]).days} days")

    # Initialize result dictionary with 1-minute data
    data_dict = {'1m': df_1m}
    
    # Resample to each target timeframe
    for tf in timeframes:
        print(f"Resampling to {tf}...")
        data_dict[tf] = resample_ohlcv(df_1m, tf)
        print(f"  -> {len(data_dict[tf]):,} bars")
    
    return data_dict

def get_data_summary(data_dict: Dict[str, pd.DataFrame]) -> dict:
    """
    Generate a summary of the loaded data for API responses.
    """
    summary = {}
    for tf, df in data_dict.items():
        summary[tf] = {
            'bars': len(df),
            'start': df.index[0].isoformat(),
            'end': df.index[-1].isoformat(),
            'duration_days': (df.index[-1] - df.index[0]).days,
        }
    return summary

# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    import time
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "XBTUSD"
    timeframes = ['5m', '1h', '4h', '1d']
    
    try:
        t0 = time.time()
        # Test with a specific lookback of 1 year to verify filtering
        data_dict = prepare_data(symbol, data_dir=".", timeframes=timeframes, lookback_years=7)
        t1 = time.time()
        
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"Execution Time: {t1 - t0:.4f} seconds")
        
        summary = get_data_summary(data_dict)
        for tf, info in summary.items():
            print(f"\n{tf}:")
            print(f"  Bars: {info['bars']:,}")
            print(f"  Period: {info['start'][:10]} to {info['end'][:10]}")
            print(f"  Duration: {info['duration_days']} days")
            
    except Exception as e:
        print(f"Error: {e}")
