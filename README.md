### Backtesting UI tool

A UI to streamline time-series analysis, as well as strategy testing. 
This tool will load 1 minute .csv timeseries data and analyze it for the following:
    - Momentum and Trend Following
    - Statistical arbitrage
    - Volatility Trading
    - Data Quality Statistics

Once loaded into DuckDB, and the higher timeframes databases are created, the 
tool should attach all results of the timeframe to the associated database. 
The test should get the past 10 years max. 
