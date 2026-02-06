import duckdb
from data_handler import load_csv_to_duckdb, prepare_data, get_data_summary, repair_data_gaps
from gate0 import run_chan_integrity_gate, perform_adf_test, calculate_hurst_exponent

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
filename = 'XBTUSD_1.csv'
symbol = 'BTC-USD'
timeframe = '1m'
requested_tfs = ['1m', '5m', '15m', '1h', '4h', '1d']

print("\n[MAIN] Loading in Data....\n\n")
load_csv_to_duckdb(filename, symbol, timeframe, con)
print("\n[MAIN] Bundling Data to Pandas Dict....\n\n")
data_bundle = prepare_data('BTC-USD', conn=con, timeframes=requested_tfs, lookback_years=2)
print("\n[MAIN] Summarizing data from data_bundle....\n\n")
summary = get_data_summary(data_bundle)
print("\nData Summary:")
print_timeframe_stats(summary)


for tf in requested_tfs:
    # Gate 0 on each tf
    print("\n[MAIN] Running gate 0 check on 1m....\n\n")
    passed = run_chan_integrity_gate(symbol, tf, conn=con, lookback_years=2)
    if not passed:
        print("\n[MAIN] Integrity check failed. Repairing data\n\n")
        repair_data_gaps(symbol, tf, con)
        run_chan_integrity_gate(symbol, tf, conn=con, lookback_years=2)

data_bundle = prepare_data('BTC-USD', conn=con, timeframes=requested_tfs, lookback_years=2)
print("\n[MAIN] Summarizing data from data_bundle....\n\n")
summary = get_data_summary(data_bundle)
print("\nData Summary:")
print_timeframe_stats(summary)

adf_result = perform_adf_test(symbol, '1h', 0.5, con)
adf2_result = perform_adf_test(symbol, '1h', 1, con)
print(f"\nADF results for {symbol} at the 1h timeframe, 6 months:")
print_dictionary_contents(adf_result)
print(f"\nADF results for {symbol} at the 1h timeframe, 12 months:")
print_dictionary_contents(adf2_result)

hurst_result = calculate_hurst_exponent(symbol, '4h', 0.5, con)
hurst2_result = calculate_hurst_exponent(symbol, '4h', 1, con)
print(f"\nHurst results for {symbol} at the 4h timeframe, 6 months:")
print_dictionary_contents(hurst_result)
print(f"\nHurst results for {symbol} at the 4h timeframe, 12 months:")
print_dictionary_contents(hurst2_result)


con.close()

