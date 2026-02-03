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

db_path = 'financial_data.duckdb'
con = duckdb.connect(db_path)

con.execute("""
    CREATE TABLE IF NOT EXISTS market_data (
        unixtime BIGINT,
        symbol VARCHAR,
        timeframe VARCHAR,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume DOUBLE,
        num_trades BIGINT,
        PRIMARY KEY (symbol, timeframe, unixtime)
    );
""")

def load_csv_to_duckdb(csv_path, symbol, timeframe, conn):
    """
    Directly ingests raw CSV data into DuckDB. 

    Args:
        csv_path: path of csv data to ingest
        symbol: symbol to store in database
        timeframe: timeframe to store in db

    """
    print(f"Loading {symbol} ({timeframe}) into DuckDB ingest...")
    
    # define the columns expected in the CSV file 
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
    
    # The Query:
    # 1. We specify the target columns in the INSERT statement to be safe.
    # 2. We Select from read_csv, injecting the 'symbol' and 'timeframe' literals on the fly.
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
    
    # Execute
    conn.execute(query)
    
    # Optional: Get count of what was actually inserted (vs ignored)
    # Note: changes() returns total rows affected by the last query
    affected = conn.fetchall() # This might return empty depending on version, usually execute is enough
    
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
    return 60 # Default to 1m

def resample_ohlcv(symbol: str, timeframe: str, conn: duckdb.DuckDBPyConnection):
    """
    Resamples 1-minute data in the database to higher timeframes.
    Uses FLOOR() to guarantee deterministic bucketing.
    """
    # Check if data exists
    exists_query = "SELECT COUNT(*) FROM market_data WHERE symbol = ? AND timeframe = ?"
    count = conn.execute(exists_query, [symbol, timeframe]).fetchone()[0]
    
    if count > 0:
        return

    print(f"Resampling {symbol} 1m -> {timeframe}...")
    
    seconds_bucket = _parse_timeframe_to_seconds(timeframe)
    
    # Perform Aggregation
    # We cast to DOUBLE to allow division, FLOOR it to kill the remainder, 
    # then multiply back to get the bucket start.
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
    Anchors the lookback window to the end of the dataset, not the current wall time.
    """
    # 1. Ensure the timeframe exists (lazy generation)
    resample_ohlcv(symbol, timeframe, conn)
    
    # 2. Find the most recent timestamp in the DB for this symbol/timeframe
    # We query this first to establish our anchor point.
    max_ts_query = "SELECT MAX(unixtime) FROM market_data WHERE symbol = ? AND timeframe = ?"
    max_ts = conn.execute(max_ts_query, [symbol, timeframe]).fetchone()[0]

    # Handle case where no data exists even after attempted resampling
    if max_ts is None:
        return pd.DataFrame()

    # 3. Calculate Start Timestamp (Epoch Seconds)
    # 365 days * 24 hours * 60 minutes * 60 seconds = 31,536,000 seconds/year
    seconds_per_year = 31_536_000
    cutoff_ts = max_ts - (lookback_years * seconds_per_year)
    
    # 4. Query Data within the Window
    query = """
        SELECT unixtime, open, high, low, close, volume, num_trades
        FROM market_data
        WHERE symbol = ? 
          AND timeframe = ? 
          AND unixtime >= ?
        ORDER BY unixtime ASC
    """
    
    # 5. Fetch as Arrow
    arrow_result = conn.execute(query, [symbol, timeframe, cutoff_ts]).arrow()
    
    # Handle RecordBatchReader vs Table (DuckDB version compatibility)
    if isinstance(arrow_result, pa.lib.RecordBatchReader):
        arrow_table = arrow_result.read_all()
    else:
        arrow_table = arrow_result

    df = arrow_table.to_pandas()
    
    if df.empty:
        return df

    # 6. Formatting for Quant usage (Datetime Index)
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
# 3. Execution / Testing
# ==========================================

# 1. Load Dummy Data (Using your 'XBTUSD_1.csv' example logic)
# Ensure you have a CSV file named 'XBTUSD_1.csv' available, or this line will fail.
load_csv_to_duckdb('XBTUSD_1.csv', 'BTC-USD', '1m', con)

# 2. Request Data (Logic Test)
# This will trigger the resample logic to create 5-minute bars from the 1-minute data
requested_tfs = ['1m', '5m', '1h']
data_bundle = prepare_data('BTC-USD', conn=con, timeframes=requested_tfs, lookback_years=7)

# 3. View Summary
summary = get_data_summary(data_bundle)
print("\nData Summary:")
print(summary)

for tf, df in data_bundle.items():
    print(f"Data for {tf}")
    print(df)

# 4. Inspect Dataframe
if '5m' in data_bundle:
    print("\nSample 5m Data (Resampled):")
    print(data_bundle['5m'].head())

con.close()
