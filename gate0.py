import duckdb
import pandas as pd
import numpy as np 
import duckdb
from statsmodels.tsa.stattools import adfuller 
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from data_handler import load_data_years, load_data_bars
from datetime import datetime, timedelta

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

def detect_volatility_regime(
    symbol: str,
    timeframe: str,
    lookback_bars: int,
    conn: duckdb.DuckDBPyConnection,
    short_window_days: float = 1.0,
    long_window_days: float = 90.0,
    expansion_threshold: float = 2.0,
    reset_threshold: float = 1.5
):
    """
    Volatility Regime Detector (Gate 0).
    
    Determines whether the current volatility environment is stable enough
    to run mean reversion or momentum analysis. Uses EWMA for fast reaction
    to volatility shocks and a rolling median baseline for robustness against
    outlier events.
    
    This function is stateless. The caution zone between reset_threshold and
    expansion_threshold is treated conservatively as a FAIL. True stateful
    hysteresis (remembering the previous regime) should be implemented at the
    orchestration layer by querying the most recent regime from DuckDB before
    calling this function.

    Parameters
    ----------
    symbol : str
        The ticker symbol to analyze.
    timeframe : str
        The candle timeframe. Supported: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
    lookback_years : float
        How many years of historical data to load from DuckDB.
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    short_window_days : float
        Number of days the short EWMA window should represent (default 1.0).
        Controls how quickly the detector reacts to volatility shocks.
    long_window_days : float
        Number of days the long baseline window should represent (default 90.0).
        Defines what "normal" volatility looks like.
    expansion_threshold : float
        Vol ratio above which the regime is classified as EXPANSION (default 2.0).
        Current vol must be this many multiples of baseline to trigger.
    reset_threshold : float
        Vol ratio below which the regime is confirmed as LOW_STABLE (default 1.5).
        Creates a buffer zone to prevent rapid toggling between states.

    Returns
    -------
    dict
        Diagnostic output containing vol measurements, regime classification,
        and gate pass/fail decision. Returns an error key if data is insufficient.
    """

    BARS_PER_DAY = {
        "1m": 1440,
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "4h": 6,
        "1d": 1,
        "1w": 1 / 7,
    }

    bars_per_day = BARS_PER_DAY.get(timeframe)
    if bars_per_day is None:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Supported: {list(BARS_PER_DAY.keys())}"
        )

    MIN_SHORT_SPAN = 5
    MIN_LONG_WINDOW = 30

    short_span = max(int(bars_per_day * short_window_days), MIN_SHORT_SPAN)
    long_window = max(int(bars_per_day * long_window_days), MIN_LONG_WINDOW)

    df_view = load_data_bars(symbol, timeframe, lookback_bars, conn)
    try:
        target_col = next(col for col in df_view.columns if col.lower() == 'close')
    except StopIteration:
        raise ValueError(f"Column 'close' not found in dataset for {symbol}.")

    price_series = df_view[target_col].copy()
    price_series.dropna(inplace=True)

    min_required = long_window + short_span + 1
    if len(price_series) < min_required:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "error": (
                f"Insufficient data. Need at least {min_required} bars "
                f"(short_span={short_span} + long_window={long_window} + 1) "
                f"for timeframe '{timeframe}', got {len(price_series)}."
            )
        }

    # LOG RETURNS
    log_returns = np.log(price_series / price_series.shift(1)).dropna()

    vol_ewma = log_returns.ewm(span=short_span, adjust=False).std()

    baseline_vol = vol_ewma.rolling(window=long_window).median()

    current_vol_val = vol_ewma.iloc[-1]
    baseline_vol_val = baseline_vol.iloc[-1]

    if pd.isna(baseline_vol_val) or baseline_vol_val == 0:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "error": (
                "Baseline volatility is zero or NaN. The price series may be "
                "flat or the long window contains insufficient data."
            )
        }

    vol_ratio = current_vol_val / baseline_vol_val

    if vol_ratio >= expansion_threshold:
        regime = "EXPANSION"
        gate_pass = False
    elif vol_ratio <= reset_threshold:
        regime = "LOW_STABLE"
        gate_pass = True
    else:
        regime = "NEUTRAL_CAUTION"
        gate_pass = False

    annual_factor = np.sqrt(bars_per_day * 365)

    output = {
        "symbol": symbol,
        "test_type": 'Volatility_Regime',
        "timeframe": timeframe,
        "short_span_bars": short_span,
        "long_window_bars": long_window,
        "short_window_represents": f"~{short_window_days} days",
        "long_window_represents": f"~{long_window_days} days",
        "expansion_threshold": expansion_threshold,
        "reset_threshold": reset_threshold,
        "current_vol_annualized": round(current_vol_val * annual_factor, 4),
        "baseline_vol_annualized": round(baseline_vol_val * annual_factor, 4),
        "vol_ratio": round(vol_ratio, 4),
        "regime": regime,
        "gate_pass": gate_pass
    }

    return output

def variance_ratio_test(
    symbol: str,
    timeframe: str,
    lookback_bars: int,
    conn: duckdb.DuckDBPyConnection,
    lags: list = None
):
    """
    Performs the Lo-MacKinlay Variance Ratio test on the log prices of a given symbol.
    Tests multiple lags and returns the z-score for each, identifying the dominant
    time scale of inefficiency.

    Parameters
    ----------
    symbol : str
        The trading symbol (e.g., 'BTC-USD').
    timeframe : str
        The candle timeframe (e.g., '1h').
    lookback_bars : int
        Number of bars to pull from the database.
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    lags : list, optional
        List of lag periods to test. Defaults to [2, 4, 8, 16, 32, 64].

    Returns
    -------
    dict
        Contains per-lag results, the dominant lag, and the regime classification.
    """

    if lags is None:
        lags = [2, 4, 8, 16, 32, 64]

    # 1. Obtain the data using the required function
    df_view = load_data_bars(symbol, timeframe, lookback_bars, conn)

    # 2. Handle the "Not a copy" constraint and Column Selection
    try:
        target_col = next(col for col in df_view.columns if col.lower() == 'close')
    except StopIteration:
        raise ValueError(f"Column 'close' not found in dataset for {symbol}.")

    price_series = df_view[target_col].copy()

    # 3. Preprocessing
    price_series.dropna(inplace=True)

    # We need enough data for the largest lag to be meaningful.
    # Rule of thumb: at least 10x the largest lag.
    min_required = max(lags) * 10
    if len(price_series) < min_required:
        return {
            "symbol": symbol,
            "error": (
                f"Insufficient data. Have {len(price_series)} bars, "
                f"need at least {min_required} for max lag {max(lags)}."
            )
        }

    # 4. Compute log prices and single-period log returns
    log_prices = np.log(price_series.values)
    log_returns = np.diff(log_prices)

    n = len(log_returns)
    mu = np.mean(log_returns)

    # 5. Run the Variance Ratio test for each lag
    lag_results = {}

    for q in lags:
        # --- Variance of 1-period returns ---
        # Unbiased estimator: denominator is (n - 1)
        sigma_1_sq = np.sum((log_returns - mu) ** 2) / (n - 1)

        # --- Variance of q-period returns ---
        # Construct overlapping q-period returns
        q_period_returns = log_prices[q:] - log_prices[:-q]
        m = len(q_period_returns)
        sigma_q_sq = np.sum((q_period_returns - q * mu) ** 2) / (m - 1)

        # --- Variance Ratio ---
        # Under the null hypothesis (random walk), VR = 1.0
        vr = sigma_q_sq / (q * sigma_1_sq)

        # --- Heteroscedasticity-consistent Z-Score (Lo-MacKinlay) ---
        # This is robust to time-varying volatility, which is critical
        # for crypto and any leveraged market.
        theta = 0.0
        sigma_sum_sq = np.sum((log_returns - mu) ** 2)
        for j in range(1, q):
            delta_j_sum = np.sum(
                (log_returns[j:] - mu) ** 2 * (log_returns[:-j] - mu) ** 2
            )
            delta_j = delta_j_sum / (sigma_sum_sq ** 2)
            theta += ((2 * (q - j)) / q) ** 2 * delta_j

        # Avoid division by zero in degenerate cases
        if theta <= 0:
            z_score = 0.0
            p_value = 1.0
        else:
            z_score = (vr - 1.0) / np.sqrt(theta)
            p_value = 2.0 * (1.0 - norm.cdf(abs(z_score)))

        lag_results[q] = {
            "variance_ratio": round(vr, 6),
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 6),
            "significant": abs(z_score) > 2.0
        }

    # 6. Identify the dominant lag (highest absolute z-score)
    dominant_lag = max(lag_results, key=lambda k: abs(lag_results[k]["z_score"]))
    dominant_z = lag_results[dominant_lag]["z_score"]

    # 7. Classify the regime based on the dominant signal
    #    z < -2.0 → Mean Reversion (variance grows slower than linear)
    #    z >  2.0 → Momentum (variance grows faster than linear)
    #    else     → Random Walk (no detectable inefficiency)
    if dominant_z < -2.0:
        regime = "MEAN_REVERSION_CANDIDATE"
    elif dominant_z > 2.0:
        regime = "MOMENTUM_CANDIDATE"
    else:
        regime = "RANDOM_WALK"

    # 8. Count how many lags show significance in each direction
    mr_count = sum(1 for v in lag_results.values() if v["z_score"] < -2.0)
    mom_count = sum(1 for v in lag_results.values() if v["z_score"] > 2.0)

    # 9. Format the output
    output = {
        "symbol": symbol,
        "test_type": "VarianceRatio_LoMacKinlay",
        "timeframe": timeframe,
        "lookback_bars": lookback_bars,
        "number_of_observations": n,
        "lags_tested": lags,
        "lag_results": lag_results,
        "dominant_lag": dominant_lag,
        "dominant_z_score": dominant_z,
        "regime": regime,
        "mean_reversion_lag_count": mr_count,
        "momentum_lag_count": mom_count,
        "any_significant": any(v["significant"] for v in lag_results.values())
    }

    return output

def perform_adf_test(symbol: str, timeframe: str, lookback_bars: int, conn: duckdb.DuckDBPyConnection):
    """
    Performs the Augmented Dickey-Fuller test on the 'close' price of a given symbol.
    """
    
    # 1. Obtain the data using the required function
    # Note: df_view is a read from the database
    df_view = load_data_bars(symbol, timeframe, lookback_bars, conn)
    
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
        'test_type': 'ADF_RawPrice',
        'test_statistic': result[0],
        'p_value': result[1],
        'lags_used': result[2],
        'number_of_observations': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05  # Assuming a 5% significance level
    }

    return output

def perform_adf_test_residuals(symbol: str, timeframe: str, lookback_bars: int, window: int, conn: duckdb.DuckDBPyConnection):
    """
    Performs the Augmented Dickey-Fuller test on the RESIDUALS of the price 
    (Close - Moving Average).
    
    This tests if the price mean-reverts to its moving average (The "Rubber Band" trade).
    """
    
    # 1. Obtain the data
    # We assume load_data returns a Pandas DataFrame
    df_view = load_data_bars(symbol, timeframe, lookback_bars, conn)
    
    # 2. Column Selection
    try:
        target_col = next(col for col in df_view.columns if col.lower() == 'close')
    except StopIteration:
        raise ValueError(f"Column 'close' not found in dataset for {symbol}.")
    
    # 3. Calculate Residuals
    # We work on a copy to preserve the original dataframe
    price_series = df_view[target_col].copy()
    
    # Calculate the Moving Average (The Anchor)
    rolling_mean = price_series.rolling(window=window).mean()
    
    # Calculate the Spread (The Residual)
    # This is the actual series we are trading in a mean-reversion strategy
    residuals = price_series - rolling_mean
    
    # 4. Preprocessing
    # The first 'window' rows will be NaN due to the rolling mean.
    residuals.dropna(inplace=True)
    
    # Ensure we have enough data points (post-windowing)
    if len(residuals) < 30: # Slightly higher requirement due to windowing loss
        return {
            "symbol": symbol,
            "error": "Insufficient data points to perform ADF on residuals."
        }

    # 5. Run ADF on Residuals
    # autolag='AIC' ensures we handle serial correlation correctly
    result = adfuller(residuals, autolag='AIC')

    # 6. Format the output
    # Note: Per your specific logic, we look for p < 0.01 for residuals
    output = {
        'symbol': symbol,
        'test_type': 'ADF_Residuals',
        'ma_window': window,
        'test_statistic': result[0],
        'p_value': result[1],
        'lags_used': result[2],
        'number_of_observations': result[3],
        'critical_values': result[4],
        # Stricter threshold for Mean Reversion candidates (p < 0.01)
        'is_stationary': result[1] < 0.01 
    }

    return output

def calculate_hurst_exponent(symbol: str, timeframe: str, lookback_bars: int, conn: duckdb.DuckDBPyConnection) -> dict:
    """
    Calculates the Hurst Exponent (H) using Rescaled Range (R/S) Analysis.
    Optimized for the 'Scientific Market Approach' pipeline.
    """
    
    # 1. Obtain Data (Assumes load_data handles the SQL query internally)
    # Ensure your load_data function filters by date using lookback_bars
    df = load_data_bars(symbol, timeframe, lookback_bars, conn)
    
    # 2. Critical Data Validation
    if df.empty or len(df) < 200:
        # Scientific Rule: R/S analysis is statistically unstable < 200 data points
        print(f"Skipping {symbol}: Insufficient data points ({len(df)}) for robust Hurst.")
        return {"symbol": symbol, "hurst_exponent": 0.5, "regime": "INSUFFICIENT_DATA"}

    if 'Close' not in df.columns:
        col_map = {c.lower(): c for c in df.columns}
        if 'close' in col_map:
            df = df.rename(columns={col_map['close']: 'Close'})
        else:
            raise KeyError(f"Column 'Close' not found in {symbol} data.")

    # 3. Pre-processing
    # We use Log Returns for R/S analysis to determine the character of the price path
    prices = df['Close'].astype(float).values
    log_prices = np.log(prices)
    log_returns = np.diff(log_prices)
    
    # Clean Data
    series = log_returns[~np.isnan(log_returns) & ~np.isinf(log_returns)]
    size = len(series)

    # 4. Rescaled Range (R/S) Analysis
    # We use powers of 2 for lags to ensure logarithmic spacing (better for polyfit)
    min_lag = 10
    max_lag = size // 2
    
    # Create logarithmically spaced lags
    lags = np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag), num=20).astype(int))
    lags = lags[lags > min_lag] # Ensure strictly > min_lag
    
    rs_values = []
    
    for lag in lags:
        # Split series into non-overlapping chunks
        num_chunks = size // lag
        if num_chunks < 1: continue

        rs_for_lag = []
        
        for i in range(num_chunks):
            chunk = series[i*lag : (i+1)*lag]
            
            # R/S Calculation
            chunk_mean = np.mean(chunk)
            chunk_cumsum = np.cumsum(chunk - chunk_mean) # Deviations
            
            r_val = np.max(chunk_cumsum) - np.min(chunk_cumsum) # Range
            s_val = np.std(chunk, ddof=1) # Standard Deviation
            
            if s_val > 0:
                rs_for_lag.append(r_val / s_val)
        
        if rs_for_lag:
            rs_values.append(np.mean(rs_for_lag))
        else:
            rs_values.append(np.nan)

    # 5. Calculate H via Polyfit (No sklearn dependency)
    # Filter valid entries
    valid_mask = (np.array(rs_values) > 0)
    if np.sum(valid_mask) < 3:
        return {"symbol": symbol, "hurst_exponent": 0.5, "regime": "RANDOM_WALK"}

    y_vals = np.log(np.array(rs_values)[valid_mask])
    x_vals = np.log(lags[valid_mask])
    
    # Fit line: log(R/S) = H * log(lag) + C
    poly = np.polyfit(x_vals, y_vals, 1)
    hurst_exponent = poly[0]
    
    # 6. Interpret Result (Aligned with our Plan)
    regime = "RANDOM_WALK" # Default (0.4 <= H <= 0.6)
    
    if hurst_exponent < 0.4:
        regime = "MEAN_REVERSION"
    elif hurst_exponent > 0.6: # FIXED: Changed from 0.65 to 0.6 per plan
        regime = "MOMENTUM"

    output = {
        "symbol": symbol,
        "test_type": "Hurst_Exponent",
        "hurst_exponent": float(hurst_exponent),
        "regime": regime,
        "n_obs": size,
        "valid": True
    }
    
    return output





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

def calculate_ou_halflife(symbol: str, timeframe: str, lookback_years: float, conn: duckdb.DuckDBPyConnection) -> dict:
    """
    Calculates the Ornstein-Uhlenbeck Half-life of Mean Reversion.
    Based on the regression: (p_t - p_{t-1}) = alpha + lambda * p_{t-1} + e
    Half-life = -ln(2) / lambda
    """
    
    # 1. Obtain Data (Reusing your existing load_data helper)
    df = load_data(symbol, timeframe, lookback_years, conn)
    
    # 2. Critical Data Validation
    if df.empty:
        raise ValueError(f"Insufficient data for symbol '{symbol}'.")

    # Standardize column name to 'Close'
    if 'Close' not in df.columns:
        if 'close' in df.columns:
            df = df.rename(columns={'close': 'Close'})
        else:
            raise KeyError(f"Column 'Close' not found. Available columns: {df.columns.tolist()}")

    # 3. Pre-processing
    # We use the price series (or log prices) for OU. 
    # Log prices are generally preferred for financial time series to handle heteroscedasticity.
    z = np.log(df['Close'].astype(float))
    
    # Define the response: (p_t - p_{t-1}) -> The change in price
    # Define the predictor: p_{t-1} -> The previous price
    prev_z = z.shift(1)
    dz = z - prev_z
    
    # Drop the first row which is now NaN due to shifting
    df_reg = pd.DataFrame({'dz': dz, 'prev_z': prev_z}).dropna()

    if len(df_reg) < 30:
        raise ValueError(f"Time series too short ({len(df_reg)} points) for reliable OU estimation.")

    # 4. Linear Regression
    # dz = alpha + lambda * prev_z
    x = df_reg['prev_z'].values.reshape(-1, 1)
    y = df_reg['dz'].values
    
    model = LinearRegression()
    model.fit(x, y)
    
    lambda_val = model.coef_[0]  # This is the speed of mean reversion
    
    # 5. Calculate Half-Life
    # If lambda is positive, the series is diverging (trending/momentum), not mean reverting.
    if lambda_val >= 0:
        half_life = np.inf
        is_mean_reverting = False
    else:
        # Half-life = -ln(2) / lambda
        half_life = -np.log(2) / lambda_val
        is_mean_reverting = True

    # 6. Prepare Output
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "lambda_coefficient": float(lambda_val),
        "half_life_periods": float(half_life) if half_life != np.inf else "Infinity",
        "is_mean_reverting": is_mean_reverting,
        "n_obs": len(df_reg),
        "interpretation": f"Takes ~{round(half_life, 2) if is_mean_reverting else 'N/A'} {timeframe} periods to revert halfway to the mean."
    }


