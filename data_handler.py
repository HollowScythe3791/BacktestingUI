"""
data_handler.py
Multi-timeframe data preparation module for the VectorBT validation engine.
Optimized for memory efficiency using chunked loading.
"""

import pandas as pd
import numpy as np
import gc
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta

def get_last_timestamp(filepath: str) -> int:
    """
    Efficiently read the last line of a file to get the last timestamp
    without loading the whole file.
    """
    with open(filepath, 'rb') as f:
        try:
            f.seek(-1024, os.SEEK_END)
        except OSError:
            # File is smaller than 1024 bytes, read from beginning
            f.seek(0)
            
        last_lines = f.readlines()
        if not last_lines:
            return 0
            
        # Get the very last line
        last_line = last_lines[-1].decode('utf-8')
        
        # Parse the first column (Timestamp)
        try:
            timestamp = int(float(last_line.split(',')[0]))
            return timestamp
        except (IndexError, ValueError):
            return 0

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
    # 1. Get the last timestamp in the file to calculate the cutoff
    last_ts = get_last_timestamp(filepath)
    if last_ts == 0:
        raise ValueError(f"Could not determine valid timestamp from {filepath}")

    # Calculate cutoff (Current End - Years)
    # Kraken timestamps are in seconds
    seconds_per_year = 31536000    # 360 * 24 * 60 * 60
    cutoff_ts = last_ts - (lookback_years * seconds_per_year)
    
    print(f"Filtering data: Loading from {datetime.utcfromtimestamp(cutoff_ts)} to {datetime.utcfromtimestamp(last_ts)}")

    # 2. Read in chunks
    chunk_size = 100000 # Process 100k rows at a time
    chunks = []
    
    # Define column names
    col_names = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades']
    
    # Create iterator
    reader = pd.read_csv(
        filepath, 
        header=None, 
        names=col_names, 
        chunksize=chunk_size,
        dtype={'Timestamp': float, 'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float}
    )

    for chunk in reader:
        # Check if the chunk contains data we need
        # If the last row of the chunk is older than cutoff, skip the whole chunk
        if chunk.iloc[-1]['Timestamp'] < cutoff_ts:
            continue
        
        # If the chunk overlaps with our window
        if chunk.iloc[0]['Timestamp'] < cutoff_ts:
            # Filter rows inside this specific chunk
            chunk = chunk[chunk['Timestamp'] >= cutoff_ts]
        
        # Drop 'Trades' column immediately to save RAM
        chunk.drop('Trades', axis=1, inplace=True)
        chunks.append(chunk)

    if not chunks:
        raise ValueError("No data found within the specified lookback period.")

    # 3. Concatenate valid chunks
    df = pd.concat(chunks, ignore_index=True)
    
    # Clean up chunks list to free memory
    del chunks
    gc.collect()

    # 4. Final Formatting
    # Convert epoch timestamp to datetime index (UTC)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s', utc=True)
    df.set_index('Timestamp', inplace=True)
    
    # Sort index to ensure chronological order
    df.sort_index(inplace=True)
    
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
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "XBTUSD"
    timeframes = ['5m', '1h', '4h', '1d']
    
    try:
        # Test with a specific lookback of 1 year to verify filtering
        data_dict = prepare_data(symbol, data_dir=".", timeframes=timeframes, lookback_years=7)
        
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        
        summary = get_data_summary(data_dict)
        for tf, info in summary.items():
            print(f"\n{tf}:")
            print(f"  Bars: {info['bars']:,}")
            print(f"  Period: {info['start'][:10]} to {info['end'][:10]}")
            print(f"  Duration: {info['duration_days']} days")
            
    except Exception as e:
        print(f"Error: {e}")
