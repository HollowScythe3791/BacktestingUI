import duckdb
from data_handler import load_csv_to_duckdb, prepare_data, get_data_summary, repair_data_gaps
from gate0 import run_chan_integrity_gate, perform_adf_test, perform_adf_test_residuals, calculate_hurst_exponent, calculate_ou_halflife, detect_volatility_regime, variance_ratio_test
# Connectin to database
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def print_vr_temperature_grid(all_results: dict):
    """
    Prints a temperature grid of Variance Ratio z-scores across
    all timeframes and lags.

    Parameters
    ----------
    all_results : dict
        Keyed by timeframe string, values are the output dicts
        from variance_ratio_test().

    Example input:
        {
            "15m": vr_results_15m,
            "1h": vr_results_1h,
            "4h": vr_results_4h,
            "1d": vr_results_1d
        }
    """

    # 1. Collect all unique lags across timeframes (sorted)
    all_lags = sorted(set(
        lag
        for res in all_results.values()
        if "lag_results" in res
        for lag in res["lag_results"].keys()
    ))

    # 2. Define the color mapping based on z-score
    def colorize_z(z_score):
        """
        Maps a z-score to a colored, fixed-width string.
        
        Blue/Cyan   = Mean Reversion (z < -2)
        White/Gray  = Random Walk (-2 < z < 2)
        Red/Yellow  = Momentum (z > 2)
        """
        text = f"{z_score:>7.2f}"

        # Strong Mean Reversion
        if z_score <= -3.0:
            return f"\033[1;34m{text}\033[0m"   # Bold Blue
        elif z_score <= -2.0:
            return f"\033[36m{text}\033[0m"      # Cyan
        # Weak / Random Walk
        elif z_score < 2.0:
            return f"\033[90m{text}\033[0m"      # Gray (noise)
        # Momentum
        elif z_score < 3.0:
            return f"\033[33m{text}\033[0m"      # Yellow
        else:
            return f"\033[1;31m{text}\033[0m"    # Bold Red

    def regime_label(z_score):
        if z_score <= -3.0:
            return "MR++"
        elif z_score <= -2.0:
            return "MR"
        elif z_score < 2.0:
            return "RW"
        elif z_score < 3.0:
            return "MOM"
        else:
            return "MOM++"

    # 3. Extract the symbol name from the first result
    symbol = next(
        (res.get("symbol", "???") for res in all_results.values() if "symbol" in res),
        "???"
    )

    # 4. Print header
    print("\n" + "=" * 80)
    print(f"  VARIANCE RATIO Z-SCORE TEMPERATURE GRID — {symbol}")
    print("=" * 80)
    print()

    # Legend
    print("  Legend:")
    print(f"    \033[1;34m■ MR++\033[0m  Strong Mean Reversion (z ≤ -3.0)")
    print(f"    \033[36m■ MR\033[0m    Mean Reversion        (-3.0 < z ≤ -2.0)")
    print(f"    \033[90m■ RW\033[0m    Random Walk           (-2.0 < z < 2.0)")
    print(f"    \033[33m■ MOM\033[0m   Momentum              (2.0 ≤ z < 3.0)")
    print(f"    \033[1;31m■ MOM++\033[0m Strong Momentum       (z ≥ 3.0)")
    print()

    # 5. Build the grid
    #    Column headers (lags)
    lag_header = "  TF      │"
    for lag in all_lags:
        lag_header += f"  Lag {lag:<4}│"
    lag_header += "  Dominant"

    separator = "  " + "─" * 10 + "┼" + ("─" * 9 + "┼") * len(all_lags) + "─" * 10

    print(lag_header)
    print(separator)

    # 6. Print each timeframe row
    for tf, res in all_results.items():
        if "error" in res:
            print(f"  {tf:<8} │  ERROR: {res['error']}")
            continue

        lag_results = res.get("lag_results", {})
        dominant_lag = res.get("dominant_lag", None)
        dominant_z = res.get("dominant_z_score", 0.0)

        row = f"  {tf:<8} │"
        for lag in all_lags:
            if lag in lag_results:
                z = lag_results[lag]["z_score"]
                row += f" {colorize_z(z)} │"
            else:
                row += f"    {'—':>4}  │"

        # Dominant lag summary
        if dominant_lag is not None:
            dom_label = regime_label(dominant_z)
            row += f"  Lag {dominant_lag} ({dom_label})"

        print(row)

    print(separator)

    # 7. Print column summary (vertical consensus per lag)
    print()
    print("  Cross-Timeframe Consensus per Lag:")
    print()

    for lag in all_lags:
        z_scores = []
        for tf, res in all_results.items():
            if "lag_results" in res and lag in res["lag_results"]:
                z_scores.append(res["lag_results"][lag]["z_score"])

        if not z_scores:
            continue

        mr_count = sum(1 for z in z_scores if z <= -2.0)
        mom_count = sum(1 for z in z_scores if z >= 2.0)
        total = len(z_scores)
        avg_z = sum(z_scores) / total

        consensus = "No consensus"
        if mr_count == total:
            consensus = "\033[1;34mUNANIMOUS MEAN REVERSION\033[0m"
        elif mom_count == total:
            consensus = "\033[1;31mUNANIMOUS MOMENTUM\033[0m"
        elif mr_count > total / 2:
            consensus = "\033[36mMajority Mean Reversion\033[0m"
        elif mom_count > total / 2:
            consensus = "\033[33mMajority Momentum\033[0m"
        else:
            consensus = "\033[90mMixed / Random Walk\033[0m"

        print(f"    Lag {lag:<4}  │  Avg z: {avg_z:>6.2f}  │  MR: {mr_count}/{total}  MOM: {mom_count}/{total}  │  {consensus}")

    print()
    print("=" * 80)
# Helper functions to debug

def print_timeframe_stats(timeframe_data):
    print(f"{'TIMEFRAME':<10} | {'BARS':<10} | {'DAYS':<6} | {'LAST CLOSE':<12} | {'RANGE'}")
    print("-" * 85)
    
    for tf, info in timeframe_data.items():
        # Formatting the strings for a nice table view
        bars = f"{info['bars']:,}" # Adds thousands separator
        days = info['duration_days']
        close = f"${float(info['last_close']):,.2f}"
        start = info['start'].replace('T', ' ')
        end = info['end'].replace('T', ' ')
        
        print(f"{tf:<10} | {bars:<10} | {days:<6} | {close:<12} | {start} to {end}")

def print_dictionary_contents(data_dict):
    """
    Iterates through a dictionary and prints each key-value pair 
    followed by a newline character.
    """
    for key, value in data_dict.items():
        # The \n at the end ensures an extra space between entries
        print(f"{key}: {value}")


# load data from csv
symbol = 'BTC-USD'        # symbol to save to database
filename = f"./data/output/{symbol.replace('-','')}_1min.csv" # input file
timeframe = '1m'          # time frame of datafile to save to data base
requested_tfs = ['1m', '5m', '15m', '1h', '4h', '1d']

"""
# Summarizing successfull data load
print("\n[MAIN] Loading in Data....\n\n")
load_csv_to_duckdb(filename, symbol, timeframe, con)
print("\n[MAIN] Bundling Data to Pandas Dict....\n\n")
data_bundle = prepare_data(symbol, conn=con, timeframes=requested_tfs, lookback_years=2)
print("\n[MAIN] Summarizing data from data_bundle....\n\n")
summary = get_data_summary(data_bundle)
print("\nData Summary:")
print_timeframe_stats(summary)

# Runs integrity check on data
for tf in requested_tfs:
    # Gate 0 on each tf
    print("\n[MAIN] Running gate 0 check on 1m....\n\n")
    passed = run_chan_integrity_gate(symbol, tf, conn=con, lookback_years=2)
    if not passed:
        print("\n[MAIN] Integrity check failed. Repairing data\n\n")
        repair_data_gaps(symbol, tf, con)
        run_chan_integrity_gate(symbol, tf, conn=con, lookback_years=2)

data_bundle = prepare_data(symbol, conn=con, timeframes=requested_tfs, lookback_years=2)
print("\n[MAIN] Summarizing data from data_bundle....\n\n")
summary = get_data_summary(data_bundle)
print("\nData Summary:")
print_timeframe_stats(summary)
"""

"""
Testing gate 0:
    Checks if the data is stable, go through mean reversion and momentum tets
"""
# Data Volatility Test:
tf = '1h' # This needs to be better defined
lookback_bars = 3000
print(f"\nVolatility Regime Check for {symbol} at the {tf} timeframe, on last {lookback_bars} bars:")
regimeresult = detect_volatility_regime(symbol,tf, lookback_bars, con)
print_dictionary_contents(regimeresult)

lags = [2,4,7,16,32,64,128]
lookback_bars = 8000

timeframe_config = {
    "15m": {
        "lags": [4, 8, 16, 32, 64, 96, 128, 192],
        "lookback_bars": 2000
    },
    "1h": {
        "lags": [2, 4, 8, 12, 24, 48, 96, 168],
        "lookback_bars": 2000
    },
    "4h": {
        "lags": [2, 3, 6, 12, 24, 36, 72],
        "lookback_bars": 1000
    },
    "1d": {
        "lags": [2, 5, 10, 21, 42, 63],
        "lookback_bars": 1000
    }
}

all_vr_results = {}

for tf, config in timeframe_config.items():
    vr_results = variance_ratio_test(symbol, tf, config['lookback_bars'], con, config['lags'])
    all_vr_results[tf] = vr_results

print_vr_temperature_grid(all_vr_results)

"""
tfs = ['15m','1h', '4h', '1d']
if regimeresult['gate_pass']: # Get results from volatility tests

    # Mean Reversion Test:

    lookback_bars = 300

    for tf in tfs:
        adf_result = perform_adf_test(symbol, tf, lookback_bars, con)
        print(f"\nADF results for {symbol} at the {tf} timeframe, on last {lookback_bars} bars:")
        print_dictionary_contents(adf_result)

        adfr_result = perform_adf_test_residuals(symbol, tf, lookback_bars, 50, con)
        print(f"\nADF results on residuals for {symbol} at the {tf} timeframe, on last {lookback_bars} bars:")
        print_dictionary_contents(adfr_result)

        hurst_result = calculate_hurst_exponent(symbol, tf, lookback_bars, con)
        print(f"\nHurst results on residuals for {symbol} at the {tf} timeframe, on last {lookback_bars} bars:")
        print_dictionary_contents(hurst_result)


    # Momentum Test:
"""

con.close()

