import duckdb
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from data_handler import load_data

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
                            lookback_years: float = 2) -> bool:
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


import pandas as pd
import duckdb
from statsmodels.tsa.stattools import adfuller

def perform_adf_test(symbol: str, timeframe: str, lookback_years: float , conn: duckdb.DuckDBPyConnection):
    """
    Performs the Augmented Dickey-Fuller test on the 'close' price of a given symbol.
    """
    
    # 1. Obtain the data using the required function
    # Note: df_view is a read from the database
    df_view = load_data(symbol, timeframe, lookback_years, conn)
    
    # 2. Handle the "Not a copy" constraint and Column Selection
    # We look for a 'close' column (case-insensitive) and explicitly .copy() 
    # the series to detach it from the database view. This allows us to 
    # drop NaNs and manipulate data without affecting the underlying DuckDB object.
    try:
        target_col = next(col for col in df_view.columns if col.lower() == 'close')
    except StopIteration:
        raise ValueError(f"Column 'close' not found in dataset for {symbol}.")
        
    price_series = df_view[target_col].copy()
    
    # 3. Preprocessing
    # The ADF test cannot handle NaN values.
    price_series.dropna(inplace=True)
    
    # Ensure we have enough data points to run the test
    if len(price_series) < 15:
        return {
            "symbol": symbol,
            "error": "Insufficient data points to perform Augmented Dickey-Fuller test."
        }

    # 4. Run the Augmented Dickey-Fuller test
    # Result tuple: (adf_stat, p_value, used_lag, nobs, critical_values, icbest)
    result = adfuller(price_series)

    # 5. Format the output
    output = {
        'symbol': symbol,
        'test_statistic': result[0],
        'p_value': result[1],
        'lags_used': result[2],
        'number_of_observations': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05  # Assuming a 5% significance level
    }

    return output



def calculate_hurst_exponent(symbol: str, timeframe: str, lookback_years: float, conn: duckdb.DuckDBPyConnection) -> dict:
    """
    Calculates the Hurst Exponent (H) using Rescaled Range (R/S) Analysis.
    
    Parameters:
    - symbol: Ticker symbol (e.g., 'BTC-USD')
    - timeframe: Bar granularity (e.g., '1h')
    - lookback_years: Depth of history (Float allowed, e.g., 0.5)
    - conn: Active DuckDB connection
    
    Returns:
    - Dictionary containing the Hurst Exponent and interpretation data.
    """
    
    # 1. Obtain Data
    # Ensure your load_data function casts the calculated timestamp to int!
    df = load_data(symbol, timeframe, lookback_years, conn)
    
    # 2. Critical Data Validation
    if df.empty:
        # Detailed error to help debug the "Insufficient Data" traceback
        raise ValueError(
            f"Insufficient data for symbol '{symbol}'. "
            f"load_data returned 0 rows for timeframe '{timeframe}' and lookback {lookback_years}. "
            f"Please verify that '{symbol}' exists in the 'market_data' table and matches the case exactly."
        )

    if 'Close' not in df.columns:
        # Handle potential case sensitivity in column names
        if 'close' in df.columns:
            df = df.rename(columns={'close': 'Close'})
        else:
            raise KeyError(f"Column 'Close' not found. Available columns: {df.columns.tolist()}")

    # 3. Pre-processing
    # Detach from DB view and drop NaNs
    price_series = df['Close'].dropna()
    
    # Check length requirements for statistical significance
    if len(price_series) < 100:
        raise ValueError(f"Time series too short ({len(price_series)} bars). Hurst requires > 100 data points.")

    # Log Prices (Financial time series are generally log-normally distributed)
    prices = np.log(price_series.copy().to_numpy())
    size = len(prices)

    # 4. Rescaled Range (R/S) Analysis
    # We define the minimum lag (chunk size) and maximum lag.
    min_lag = 10
    max_lag = size // 2
    lags = range(min_lag, max_lag, 5) 
    
    rs_values = []
    
    for lag in lags:
        # Calculate number of chunks for this lag size
        num_chunks = size // lag
        rs_for_lag = []
        
        for i in range(num_chunks):
            start = i * lag
            end = start + lag
            chunk = prices[start:end]
            
            # Calculate R/S for this chunk
            chunk_mean = np.mean(chunk)
            chunk_centered = chunk - chunk_mean # Deviations from mean
            cumulative_deviations = np.cumsum(chunk_centered)
            
            # Range (R)
            r_val = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            
            # Standard Deviation (S)
            s_val = np.std(chunk, ddof=1)
            
            # Avoid division by zero (if price was constant)
            if s_val == 0:
                rs_for_lag.append(0)
            else:
                rs_for_lag.append(r_val / s_val)
        
        # Average R/S for this lag size
        if len(rs_for_lag) > 0:
            rs_values.append(np.mean(rs_for_lag))
        else:
            rs_values.append(np.nan)

    # 5. Calculate H via Linear Regression (Log-Log plot)
    # Filter valid R/S values
    valid_indices = [i for i, rs in enumerate(rs_values) if rs > 0]
    
    if len(valid_indices) < 3:
         raise ValueError("Could not compute valid R/S values (series might be constant or too short).")

    # y = log(R/S), x = log(lag)
    y_vals = np.log([rs_values[i] for i in valid_indices])
    x_vals = np.log([lags[i] for i in valid_indices]).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(x_vals, y_vals)
    
    hurst_exponent = model.coef_[0]
    
    # 6. Interpret Result
    if hurst_exponent < 0.45:
        regime = "Mean Reverting"
    elif hurst_exponent > 0.55:
        regime = "Trending"
    else:
        regime = "Random Walk"

    output = {
        "symbol": symbol,
        "hurst_exponent": float(hurst_exponent),
        "regime": regime,
        "n_obs": size,
        "lookback_years": lookback_years,
        "interpretation": "H < 0.5: Mean Reverting | H = 0.5: Random Walk | H > 0.5: Trending"
    }
    
    return output
