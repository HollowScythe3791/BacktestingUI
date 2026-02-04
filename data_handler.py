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

def run_chan_integrity_gate(symbol, timeframe, lookback_years=2, db_path='financial_data.duckdb'):
    """
    Runs integrity check on past lookback_years of data. 
    Includes: Window Density, Amihud Ratio, and Time-Step Integrity.
    """

    con = duckdb.connect(db_path)
    print(f"--- 🛡️ Gate 0: Integrity Report for {symbol} ({timeframe}) | Lookback: {lookback_years}y ---")
    
    # 1. Determine the Time Window
    max_ts = con.execute("SELECT MAX(unixtime) FROM market_data WHERE symbol = ? AND timeframe = ?", [symbol, timeframe]).fetchone()[0]
    if max_ts is None:
        print("❌ ERROR: Database is empty.")
        return False
    
    cutoff_ts = max_ts - (lookback_years * 31_536_000)
    step = _parse_timeframe_to_seconds(timeframe)

    # 2. Density Check within Window
    density_query = """
    SELECT 
        COUNT(*) as actual,
        ((MAX(unixtime) - MIN(unixtime)) / ?) + 1 as expected
    FROM market_data 
    WHERE symbol = ? AND timeframe = ? AND unixtime >= ?
    """
    actual, expected = con.execute(density_query, [step, symbol, timeframe, cutoff_ts]).fetchone()
    density = round((actual / expected) * 100, 2) if expected else 0

    # 3. Amihud Ratio within Window
    amihud_query = """
    WITH rets AS (
        SELECT ABS(ln(close / LAG(close) OVER (ORDER BY unixtime))) / NULLIF(volume * close, 0) as impact
        FROM market_data 
        WHERE symbol = ? AND timeframe = ? AND unixtime >= ?
    )
    SELECT COALESCE(AVG(impact), 0) FROM rets WHERE impact IS NOT NULL
    """
    amihud_val = con.execute(amihud_query, [symbol, timeframe, cutoff_ts]).fetchone()[0]

    # 4. Time-Step Integrity Check
    # This checks if the difference between consecutive rows equals the expected step.
    step_integrity_query = """
    WITH steps AS (
        SELECT 
            unixtime,
            LEAD(unixtime) OVER (ORDER BY unixtime) - unixtime as step_diff
        FROM market_data
        WHERE symbol = ? AND timeframe = ? AND unixtime >= ?
    )
    SELECT 
        COUNT(*) as violations,
        MAX(step_diff) as max_gap_seconds
    FROM steps
    WHERE step_diff IS NOT NULL AND step_diff != ?
    """
    violations, max_gap = con.execute(step_integrity_query, [symbol, timeframe, cutoff_ts, step]).fetchone()
    violations = violations if violations else 0
    max_gap = max_gap if max_gap else 0

    # --- VERDICT ---
    # We pass if density is high AND there are no step violations
    # (Note: For crypto 24/7 markets, violations should be 0. For stocks, weekends will cause violations).
    is_continuous = (violations == 0)
    status = "✅ PASS" if density > 99.0 and is_continuous else "❌ FAIL"
    
    print(f"📊 Window Density: {density}%")
    print(f"⏱️ Time-Step Violations: {violations} (Max Gap: {max_gap}s)")
    print(f"💧 Window Amihud: {amihud_val:.12f}")
    print(f"⚖️ Final Verdict: {status}")
    
    if not is_continuous:
        print(f"   -> Found {violations} gaps where time difference != {step}s.")
    
    if density <= 99.0:
        print("💡 Suggestion: Run repair_data_and_fill_gaps() to fix the missing bars.")
    
    con.close()
    return status == "✅ PASS"

# ==========================================
# Execution Example
# ==========================================

            

def repair_data_gaps(symbol: str, timeframe: str, db_path='financial_data.duckdb'):
    """
    Identifies gaps in the time series and fills them with 
    flat candles (previous close) and 0 volume.
    """
    con = duckdb.connect(db_path)
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
    
    con.execute(repair_query)
    print(f"✅ Repair complete. Gaps filled with 0-volume flat bars.")
    
    con.close()

# ==========================================
# HOW TO RUN THE FIX
# ==========================================


# 1. Load Dummy Data (Using your 'XBTUSD_1.csv' example logic)
# Ensure you have a CSV file named 'XBTUSD_1.csv' available.
try:
    load_csv_to_duckdb('XBTUSD_1.csv', 'BTC-USD', '1m', con)
except Exception as e:
    print(f"Skipping CSV load (File not found or error): {e}")

# 2. Request Data (Logic Test)
requested_tfs = ['1m', '5m', '1h']
data_bundle = prepare_data('BTC-USD', conn=con, timeframes=requested_tfs, lookback_years=7)

# 3. View Summary
summary = get_data_summary(data_bundle)
print("\nData Summary:")
print(summary)

# 4. Run the Gate
db_file = 'financial_data.duckdb'
target_symbol = 'BTC-USD'
target_tf = '1h'

is_passed = run_chan_integrity_gate(target_symbol, target_tf, lookback_years=2)

# 1. Run the repair function
repair_data_gaps('BTC-USD', '1h')

# 2. Run the Integrity Gate again to verify PASS
print("\n--- Re-Verifying Integrity ---")
run_chan_integrity_gate('BTC-USD', '1h', lookback_years=2)
con.close()

