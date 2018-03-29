import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
from random import randint


# GBM model

# Predefined Functions
def load_price_from_coinmarketcap(coin, start, end):
    bitcoin_market_info = \
    pd.read_html("https://coinmarketcap.com/currencies/" + coin + "/historical-data/?start=" + start + "&end=" + end)[0]
    # convert the date string to the correct date format
    bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
    # when Volume is equal to '-' convert it to 0
    # bitcoin_market_info.loc[bitcoin_market_info['Volume'] == "-", 'Volume'] = 0
    # convert to int
    bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')
    # look at the first few rows
    return bitcoin_market_info


def daily_return(adj_close):
    returns = []
    for i in range(0, len(adj_close) - 1):
        today = adj_close[i ]
        yesterday = adj_close[i + 1]
        daily_return = (today - yesterday) / yesterday
        returns.append(daily_return)
    return returns


def Brownian(seed, N, T):
    np.random.seed(seed)
    dt = T / N  # time step
    b = np.random.normal(0., T, int(N)) * np.sqrt(dt)  # brownian increments
    W = np.cumsum(b)  # brownian path
    return W, b


def GBM(So, mu, sigma, W, T, N):
    t = np.linspace(0., T, N + 1)
    S = []
    S.append(So)
    for i in range(1, int(N + 1)):
        drift = (mu - 0.5 * sigma ** 2) * t[i]
        diffusion = sigma * W[i - 1]
        S_temp = So * np.exp(drift + diffusion)
        S.append(S_temp)
    return S, t


def MJD(So, mu, sigma, T, N, labda, nu, delta) -> object:
    S = []
    S.append(So)
    t = T/N
    for i in range(1, int(N+1)):
        P=np.random.poisson(labda*t)
        U=np.exp(P*nu+np.sqrt(P)*delta*np.random.normal(0,1))
        S_temp=S[i-1]*np.exp((mu-labda*(np.exp(nu+0.5*delta**2)-1)-0.5*sigma**2)*t + sigma*np.sqrt(t)*np.random.normal(0,1))*U
        S.append(S_temp)
    return S


# Calibrate Mu and Sigma
aion = load_price_from_coinmarketcap("aion", "20171018", "20180324")

writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')
aion.to_excel(writer, sheet_name="Sheet1")
writer.save()


adj_close = aion['Close']
time = np.linspace(1, len(adj_close), len(adj_close))

returns = daily_return(adj_close)

mu = np.mean(returns) * 365.  # drift coefficient
sig = np.std(returns) * np.sqrt(365.)  # diffusion coefficient
print(mu, sig)

# GBM Exact Solution
M=1000
seed = pd.Series(range(1,M+1))


So = 2.86  # Initial AION coin price (03/05/2018)
T = 1
N = 365.
times = pd.date_range('2018-03-25',periods= T*N+1, freq='D')
W = Brownian(0, N, T)[0]
# result = pd.DataFrame(GBM(So, mu, sig, W, T, N)[0])
result = pd.DataFrame()
for i in range(M):
    W = Brownian(seed[i], N, T)[0]
    soln = pd.Series(GBM(So, mu, sig, W, T, N)[0])  # Exact solution
    result = result.append(soln,ignore_index=True)
result = result.transpose()
result.index = times
print(result)
q05 = result.quantile(0.05,axis=1)
q15 = result.quantile(0.15,axis=1)
q25 = result.quantile(0.25,axis=1)
q35 = result.quantile(0.35,axis=1)
q45 = result.quantile(0.45,axis=1)
q55 = result.quantile(0.55,axis=1)
q65 = result.quantile(0.65,axis=1)
q75 = result.quantile(0.75,axis=1)
q85 = result.quantile(0.85,axis=1)
q95 = result.quantile(0.95,axis=1)
smean = result.mean(axis=1)
Stats = pd.DataFrame([smean,q05,q15,q25,q35,q45,q55,q65,q75,q85,q95])
Stats = Stats.transpose()
Stats.columns = ["mean","q05","q15","q25","q35","q45","q55","q65","q75","q85","q95"]



# Function
def GBM_simulate(coin, hist_start, hist_end, S0, Date, M):
    # Calibrate Mu and Sigma
    coin_price = load_price_from_coinmarketcap(coin, hist_start, hist_end)
    adj_close = coin_price['Close']
    time = np.linspace(1, len(adj_close), len(adj_close))
    plt.plot(time, adj_close)

    returns = daily_return(adj_close)

    mu = np.mean(returns) * 365.  # drift coefficient
    sig = np.std(returns) * np.sqrt(365.)  # diffusion coefficient
    print(mu, sig)

    # GBM Exact Solution
    seed = pd.Series(range(1, M + 1))

    T = 1
    N = 365.
    times = pd.date_range(Date, periods=T * N + 1, freq='D')
    result = pd.DataFrame()
    for i in range(M):
        W = Brownian(seed[i], N, T)[0]
        soln = pd.Series(GBM(S0, mu, sig, W, T, N)[0])  # Exact solution
        result = result.append(soln, ignore_index=True)
    result = result.transpose()
    result.index = times
    print(result)
    q05 = result.quantile(0.05, axis=1)
    q15 = result.quantile(0.15, axis=1)
    q25 = result.quantile(0.25, axis=1)
    q35 = result.quantile(0.35, axis=1)
    q45 = result.quantile(0.45, axis=1)
    q55 = result.quantile(0.55, axis=1)
    q65 = result.quantile(0.65, axis=1)
    q75 = result.quantile(0.75, axis=1)
    q85 = result.quantile(0.85, axis=1)
    q95 = result.quantile(0.95, axis=1)
    smean = result.mean(axis=1)
    Stats = pd.DataFrame([smean, q05, q15, q25, q35, q45, q55, q65, q75, q85, q95])
    Stats = Stats.transpose()
    Stats.columns = ["mean", "q05", "q15", "q25", "q35", "q45", "q55", "q65", "q75", "q85", "q95"]
    return Stats


def GBM_simulate2(mu,sig, S0, Date, M):
    # Calibrate Mu and Sigma
    # GBM Exact Solution
    seed = pd.Series(range(1, M + 1))

    T = 1
    N = 365.
    times = pd.date_range(Date, periods=T * N + 1, freq='D')
    result = pd.DataFrame()
    for i in range(M):
        W = Brownian(seed[i], N, T)[0]
        soln = pd.Series(GBM(S0, mu, sig, W, T, N)[0])  # Exact solution
        result = result.append(soln, ignore_index=True)
    result = result.transpose()
    result.index = times
    print(result)
    q05 = result.quantile(0.05, axis=1)
    q15 = result.quantile(0.15, axis=1)
    q25 = result.quantile(0.25, axis=1)
    q35 = result.quantile(0.35, axis=1)
    q45 = result.quantile(0.45, axis=1)
    q55 = result.quantile(0.55, axis=1)
    q65 = result.quantile(0.65, axis=1)
    q75 = result.quantile(0.75, axis=1)
    q85 = result.quantile(0.85, axis=1)
    q95 = result.quantile(0.95, axis=1)
    smean = result.mean(axis=1)
    Stats = pd.DataFrame([smean, q05, q15, q25, q35, q45, q55, q65, q75, q85, q95])
    Stats = Stats.transpose()
    Stats.columns = ["mean", "q05", "q15", "q25", "q35", "q45", "q55", "q65", "q75", "q85", "q95"]
    return Stats

def MJD_simulate(mu,sig, S0, Date,  labda, nu, delta, M):
    # Calibrate Mu and Sigma
    # GBM Exact Solution

    T = 2
    N = 730.
    times = pd.date_range(Date, periods=N + 1, freq='D')
    result = pd.DataFrame()
    for i in range(M):
        soln = pd.Series(MJD(S0, mu, sig, T, N, labda, nu, delta))  # Exact solution
        result = result.append(soln, ignore_index=True)
    result = result.transpose()
    result.index = times
    print(result)
    q05 = result.quantile(0.05, axis=1)
    q15 = result.quantile(0.15, axis=1)
    q25 = result.quantile(0.25, axis=1)
    q35 = result.quantile(0.35, axis=1)
    q45 = result.quantile(0.45, axis=1)
    q55 = result.quantile(0.55, axis=1)
    q65 = result.quantile(0.65, axis=1)
    q75 = result.quantile(0.75, axis=1)
    q85 = result.quantile(0.85, axis=1)
    q95 = result.quantile(0.95, axis=1)
    smean = result.mean(axis=1)
    Stats = pd.DataFrame([smean, q05, q15, q25, q35, q45, q55, q65, q75, q85, q95])
    Stats = Stats.transpose()
    Stats.columns = ["mean", "q05", "q15", "q25", "q35", "q45", "q55", "q65", "q75", "q85", "q95"]
    return Stats



Stats1 = MJD_simulate(mu=0.01458,sig=2.5849,S0=0.5,Date="20170814",labda=6.412509,nu=-0.12849,delta=0.032184,M=10000)
Stats2 = MJD_simulate(mu=0.5,sig=2.5849,S0=0.5,Date="20170814",labda=6.412509,nu=-0.12849,delta=0.032184,M=10000)
Stats3 = MJD_simulate(mu=1,sig=2.5849,S0=0.5,Date="20170814",labda=6.412509,nu=-0.12849,delta=0.032184,M=10000)
Stats4 = MJD_simulate(mu=2,sig=2.5849,S0=0.5,Date="20170814",labda=6.412509,nu=-0.12849,delta=0.032184,M=10000)
Stats5 = MJD_simulate(mu=5.390198,sig=2.5849,S0=0.5,Date="20170814",labda=6.412509,nu=-0.12849,delta=0.032184,M=10000)
Stats6 = MJD_simulate(mu=3.549762,sig=3.438839,S0=5.19,Date="20180101",labda=6.412509,nu=-0.12849,delta=0.032184,M=10000)

writer = pd.ExcelWriter('AION Project Result4.xlsx', engine='xlsxwriter')
Stats1.to_excel(writer, sheet_name="scenario1")
Stats2.to_excel(writer, sheet_name="scenario2")
Stats3.to_excel(writer, sheet_name="scenario3")
Stats4.to_excel(writer, sheet_name="scenario4")
Stats5.to_excel(writer, sheet_name="scenario5")
writer.save()

writer = pd.ExcelWriter('AION Project Result6.xlsx', engine='xlsxwriter')
Stats6.to_excel(writer, sheet_name="scenario6")
writer.save()



scenario3 = GBM_simulate2(mu=5.390198, sig=2.584968, S0=0.5, Date='2018-08-14', M=10000)
scenario4 =
writer = pd.ExcelWriter('AION Project Result3.xlsx', engine='xlsxwriter')
Stats1.to_excel(writer, sheet_name="scenario1")
Stats2.to_excel(writer, sheet_name="scenario2")
writer.save()

writer = pd.ExcelWriter('AION Project Result3.1.xlsx', engine='xlsxwriter')
Stats3.to_excel(writer, sheet_name="scenario1")
writer.save()


result = MJD(So=0.5, mu=5.390198, sigma=2.5849, T=2, N=730, labda=6.412509, nu=0.12849, delta=0.032184)
results = pd.DataFrame(result)


writer = pd.ExcelWriter('temp_obs.xlsx', engine='xlsxwriter')
results.to_excel(writer, sheet_name="scenario1")
writer.save()



TT = pd.date_range("20170814", periods=731, freq='D')
time = np.linspace(1, len(result), len(result))

plt.plot(time,result)