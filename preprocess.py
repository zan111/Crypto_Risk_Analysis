import pandas as pd
from fetch_data import worker

#correlation Correlation = direction + consistency --- Do they move together?

# Examples:

#Scenario   Correlation
#Always up/down together    ~1.0
#Opposite moves ~-1.0
#Random relation~0

#covariance When BTC moves, how big are ETH moves? -- adds magnitute to correlation 

#Variance = Covariance With Itself  Measures BTC volatility.
#var(BTC) = cov(BTC, BTC)

#Beta Beta standardizes covariance.
#beta_alt_btc = cov(alt, btc) / var(btc)

def calculate_betas(df_coindata, benchmark="BTC"):
    """
    Calculate beta of each asset vs benchmark.
    
    Parameters:
        returns_df (pd.DataFrame): Log returns
        benchmark (str): Benchmark column name
        
    Returns:
        pd.Series: Beta values
    """
    
    cov_matrix = df_coindata.cov()
    benchmark_var = cov_matrix.loc[benchmark, benchmark]
    
    betas = cov_matrix[benchmark] / benchmark_var
    
    return betas.drop(benchmark)


def build_returns_df(coin_data, use_log=True):

    returns_df = pd.DataFrame()

    for coin, df in coin_data.items():

        if use_log:
            returns_df[coin] = df["log_returns"]
        else:
            returns_df[coin] = df["returns"]

    return returns_df.dropna()

def calculate_covariance(returns_df):
    return returns_df.cov()

def calculate_correlation(returns_df):
    return returns_df.corr()
