### Backtesting UI tool

A UI to streamline time-series analysis, as well as strategy testing. 
This tool will load 1 minute .csv timeseries data and analyze it for the following:
    - Momentum and Trend Following
    - Statistical arbitrage
    - Volatility Trading
    - Data Quality Statistics

Once loaded into DuckDB, and the higher timeframes databases are created, the 
tool should attach all results of the timeframe to the associated database. 

This gives a full analyzed database to use with vectorbt. Once your strategy is
created, it will go through a multi gate testing system.
Each gate give pass/fail results and really tests the strategy on market edges.



### Usage

The project is built using python 3.10.19, though any 3.10 version should be 
stable (Version compatibility wont be fully tested until project end).

Install libraries:
```
pip install -r requirements.txt
```

WIP....

