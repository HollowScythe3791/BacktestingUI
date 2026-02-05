import duckdb
import pandas as pd

def _parse_timeframe_to_seconds(tf: str) -> int:
    """
    Helper to parse '1m', '1h', '1d' into seconds.
    Essential for defining expected time steps.
    """
    unit = tf[-1].lower()
    val = int(tf[:-1])
    if unit == 's': return val
    if unit == 'm': return val * 60
    if unit == 'h': return val * 3600
    if unit == 'd': return val * 86400
    raise ValueError(f"Unknown timeframe format: {tf}")

def _get_analysis_window(con, symbol, timeframe, lookback_years):
    """
    Determines the start (cutoff) and end (max) timestamps for the analysis.
    """
    max_ts = con.execute(
        "SELECT MAX(unixtime) FROM market_data WHERE symbol = ? AND timeframe = ?", 
        [symbol, timeframe]
    ).fetchone()[0]
    
    if max_ts is None:
        return None, None, None

    step = _parse_timeframe_to_seconds(timeframe)
    cutoff_ts = max_ts - (lookback_years * 31_536_000)
    
    return cutoff_ts, max_ts, step

def _check_window_density(con, symbol, timeframe, cutoff_ts, step):
    """
    Calculates the ratio of actual bars vs expected bars.
    """
    query = """
    SELECT 
        COUNT(*) as actual,
        ((MAX(unixtime) - MIN(unixtime)) / ?) + 1 as expected
    FROM market_data 
    WHERE symbol = ? AND timeframe = ? AND unixtime >= ?
    """
    actual, expected = con.execute(query, [step, symbol, timeframe, cutoff_ts]).fetchone()
    
    if not expected or expected == 0:
        return 0.0
        
    return round((actual / expected) * 100, 2)

def _calculate_amihud_liquidity(con, symbol, timeframe, cutoff_ts):
    """
    Calculates the Amihud Illiquidity ratio (Absolute Return / Dollar Volume).
    High values imply low liquidity (high price impact).
    """
    query = """
    WITH rets AS (
        SELECT ABS(ln(close / LAG(close) OVER (ORDER BY unixtime))) / NULLIF(volume * close, 0) as impact
        FROM market_data 
        WHERE symbol = ? AND timeframe = ? AND unixtime >= ?
    )
    SELECT COALESCE(AVG(impact), 0) FROM rets WHERE impact IS NOT NULL
    """
    return con.execute(query, [symbol, timeframe, cutoff_ts]).fetchone()[0]

def _check_timestep_integrity(con, symbol, timeframe, cutoff_ts, step):
    """
    Checks for gaps in the time series where the difference between rows
    is not equal to the timeframe step.
    """
    query = """
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
    violations, max_gap = con.execute(query, [symbol, timeframe, cutoff_ts, step]).fetchone()
    return (violations if violations else 0), (max_gap if max_gap else 0)

def run_chan_integrity_gate(symbol: str, timeframe: str, conn: duckdb.DuckDBPyConnection, 
                            lookback_years: int = 2) -> bool:
    """
    Master Orchestrator: Runs integrity checks on past lookback_years of data. 
    Aggregates results from modular sub-functions.
    """
    print(f"--- 🛡️ Gate 0: Integrity Report for {symbol} ({timeframe}) | Lookback: {lookback_years}y ---")
    
    # 1. Setup Window
    cutoff_ts, max_ts, step = _get_analysis_window(conn, symbol, timeframe, lookback_years)
    
    if max_ts is None:
        print("❌ ERROR: Database is empty or symbol not found.")
        conn.close()
        return False

    # 2. Run Modular Tests
    density = _check_window_density(conn, symbol, timeframe, cutoff_ts, step)
    amihud_val = _calculate_amihud_liquidity(conn, symbol, timeframe, cutoff_ts)
    violations, max_gap = _check_timestep_integrity(conn, symbol, timeframe, cutoff_ts, step)


    # 3. Verdict Logic
    # For crypto (24/7), violations should be 0. For TradFi, weekends/holidays naturally create gaps 
    # unless you filter for market hours beforehand. Assuming Crypto/Forex continuous here.
    is_continuous = (violations == 0)
    
    # Strict Chan Standard: We want > 99% density and perfect continuity for time-series analysis
    passed = (density > 99.0 and is_continuous)
    status = "✅ PASS" if passed else "❌ FAIL"
    
    # 4. Report
    print(f"📊 Window Density: {density}%")
    print(f"⏱️ Time-Step Violations: {violations} (Max Gap: {max_gap}s)")
    print(f"💧 Window Amihud: {amihud_val:.12f}")
    print(f"⚖️ Final Verdict: {status}")
    
    if not is_continuous:
        print(f"   -> Found {violations} gaps where time difference != {step}s.")
    
    if density <= 99.0:
        print("💡 Suggestion: Run repair_data_gaps() to fix the missing bars.")
    
    return passed


