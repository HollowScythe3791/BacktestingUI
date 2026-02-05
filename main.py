import duckdb
from data_handler import load_csv_to_duckdb, prepare_data, get_data_summary, repair_data_gaps
from gate0 import run_chan_integrity_gate

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
con.close()
