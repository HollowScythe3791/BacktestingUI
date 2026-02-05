import duckdb
import pyarrow as pa
import pandas as pd
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
# ==========================================
# 1. Setup & Existing Loading Logic
# ==========================================

def load_csv_to_duckdb(csv_path, symbol, timeframe, conn):
    """
    Directly ingests raw CSV data into DuckDB. 
    """
    print(f"Loading {symbol} ({timeframe}) into DuckDB ingest...")
    
    csv_schema = {
        'unixtime': 'BIGINT',
        'open': 'DOUBLE',
        'high': 'DOUBLE',
        'low': 'DOUBLE',
        'close': 'DOUBLE',
        'volume': 'DOUBLE',
        'num_trades': 'BIGINT'
    }
    columns_str = ", ".join([f"'{k}': '{v}'" for k, v in csv_schema.items()])
    
    query = f"""
        INSERT OR IGNORE INTO market_data (
            unixtime, symbol, timeframe, open, high, low, close, volume, num_trades
        )
        SELECT 
            unixtime, 
            '{symbol}' AS symbol, 
            '{timeframe}' AS timeframe, 
            open, 
            high, 
            low, 
            close, 
            volume, 
            num_trades
        FROM read_csv('{csv_path}', 
            header=False, 
            columns={{{columns_str}}}
        )
    """
    
    conn.execute(query)
    print(f"Ingestion complete for {symbol}.")

def _parse_timeframe_to_seconds(tf: str) -> int:
    """
    Helper to convert '5m', '1h', '1d' into seconds for SQL math.
    """
    match = re.match(r"(\d+)([mhd])", tf)
    if not match:
        raise ValueError(f"Invalid timeframe format: {tf}")
    
    val, unit = int(match.group(1)), match.group(2)
    
    if unit == 'm': return val * 60
    if unit == 'h': return val * 3600
    if unit == 'd': return val * 86400
    return 60 

def resample_ohlcv(symbol: str, timeframe: str, conn: duckdb.DuckDBPyConnection):
    """
    Resamples 1-minute data in the database to higher timeframes.
    """
    exists_query = "SELECT COUNT(*) FROM market_data WHERE symbol = ? AND timeframe = ?"
    count = conn.execute(exists_query, [symbol, timeframe]).fetchone()[0]
    
    if count > 0:
        return

    print(f"Resampling {symbol} 1m -> {timeframe}...")
    
    seconds_bucket = _parse_timeframe_to_seconds(timeframe)
    
    query = f"""
        INSERT OR IGNORE INTO market_data
        SELECT
            CAST(FLOOR(unixtime::DOUBLE / {seconds_bucket}) * {seconds_bucket} AS BIGINT) as unixtime,
            symbol,
            '{timeframe}' as timeframe,
            arg_min(open, unixtime) as open,
            max(high) as high,
            min(low) as low,
            arg_max(close, unixtime) as close,
            sum(volume) as volume,
            sum(num_trades) as num_trades
        FROM market_data
        WHERE symbol = ? AND timeframe = '1m'
        GROUP BY 1, 2
    """
    
    conn.execute(query, [symbol])
    print(f"Resampling complete for {symbol} {timeframe}.")

def load_data(symbol: str, timeframe: str, lookback_years: int, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load OHLCV data from DuckDB relative to the LAST available data point.
    """
    resample_ohlcv(symbol, timeframe, conn)
    
    max_ts_query = "SELECT MAX(unixtime) FROM market_data WHERE symbol = ? AND timeframe = ?"
    max_ts = conn.execute(max_ts_query, [symbol, timeframe]).fetchone()[0]

    if max_ts is None:
        return pd.DataFrame()

    seconds_per_year = 31_536_000
    cutoff_ts = max_ts - (lookback_years * seconds_per_year)
    
    query = """
        SELECT unixtime, open, high, low, close, volume, num_trades
        FROM market_data
        WHERE symbol = ? 
          AND timeframe = ? 
          AND unixtime >= ?
        ORDER BY unixtime ASC
    """
    
    arrow_result = conn.execute(query, [symbol, timeframe, cutoff_ts]).arrow()
    
    if isinstance(arrow_result, pa.lib.RecordBatchReader):
        arrow_table = arrow_result.read_all()
    else:
        arrow_table = arrow_result

    df = arrow_table.to_pandas()
    
    if df.empty:
        return df

    df['datetime'] = pd.to_datetime(df['unixtime'], unit='s')
    df.set_index('datetime', inplace=True)
    df.drop(columns=['unixtime'], inplace=True)
    
    return df

def prepare_data(symbol: str, conn: duckdb.DuckDBPyConnection, timeframes: Optional[List[str]] = None, lookback_years: int = 2) -> Dict[str, pd.DataFrame]:
    """
    Orchestrator to get a dictionary of dataframes for multiple timeframes.
    """
    if timeframes is None:
        timeframes = ['1m']
        
    data_store = {}
    
    for tf in timeframes:
        df = load_data(symbol, tf, lookback_years, conn)
        if not df.empty:
            data_store[tf] = df
        else:
            print(f"Warning: No data found for {symbol} {tf} within lookback window.")
            
    return data_store

def get_data_summary(data_dict: Dict[str, pd.DataFrame]) -> dict:
    """
    Generate a summary of the loaded data for API responses.
    """
    summary = {}
    for tf, df in data_dict.items():
        if df.empty:
            summary[tf] = "Empty DataFrame"
            continue
            
        summary[tf] = {
            'bars': len(df),
            'start': df.index[0].isoformat(),
            'end': df.index[-1].isoformat(),
            'duration_days': (df.index[-1] - df.index[0]).days,
            'last_close': df['close'].iloc[-1]
        }
    return summary


# ==========================================
# Execution Example
# ==========================================

def repair_data_gaps(symbol: str, timeframe: str, conn: duckdb.DuckDBPyConnection):
    """
    Identifies gaps in the time series and fills them with 
    flat candles (previous close) and 0 volume.
    """
    
    print(f"🔧 Starting Repair Process for {symbol} {timeframe}...")

    step = _parse_timeframe_to_seconds(timeframe)
    
    print("   -> Calculating missing intervals and patching...")
    
    repair_query = f"""
    WITH stats AS (
        SELECT MIN(unixtime) as min_ts, MAX(unixtime) as max_ts 
        FROM market_data 
        WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
    ),
    ideal_timeline AS (
        -- FIX: Use unnest() to turn the array of timestamps into rows
        SELECT unnest(generate_series(min_ts, max_ts, {step})) as ts
        FROM stats
    ),
    joined_data AS (
        SELECT 
            t.ts,
            m.close,
            m.unixtime
        FROM ideal_timeline t
        LEFT JOIN market_data m 
            ON t.ts = m.unixtime 
            AND m.symbol = '{symbol}' 
            AND m.timeframe = '{timeframe}'
    ),
    filled_data AS (
        SELECT
            ts,
            -- Forward fill: grab the last non-null close price
            LAST_VALUE(close IGNORE NULLS) OVER (ORDER BY ts) as fill_price,
            unixtime
        FROM joined_data
    )
    INSERT INTO market_data (unixtime, symbol, timeframe, open, high, low, close, volume, num_trades)
    SELECT 
        ts as unixtime,
        '{symbol}' as symbol,
        '{timeframe}' as timeframe,
        fill_price as open,
        fill_price as high,
        fill_price as low,
        fill_price as close,
        0 as volume,
        0 as num_trades
    FROM filled_data
    WHERE unixtime IS NULL
    """
    
    conn.execute(repair_query)
    print(f"✅ Repair complete. Gaps filled with 0-volume flat bars.")
    
