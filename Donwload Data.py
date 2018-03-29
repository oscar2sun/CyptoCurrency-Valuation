import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
from random import randint


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


aion = load_price_from_coinmarketcap("aion", "20171018", "20180324")
eth = load_price_from_coinmarketcap("ethereum", "20150807", "20180324")
eos = load_price_from_coinmarketcap("eos", "20170701", "20180324")

writer = pd.ExcelWriter('AION Project.xlsx', engine='xlsxwriter')
aion.to_excel(writer, sheet_name="aion")
eth.to_excel(writer, sheet_name="eth")
eos.to_excel(writer, sheet_name="eos")
writer.save()

# Competing Projects
btc = load_price_from_coinmarketcap("bitcoin", "20140501", "20180324")
lsk = load_price_from_coinmarketcap("lisk", "20160406", "20180324")
waves = load_price_from_coinmarketcap("waves", "20160602", "20180324")
strat = load_price_from_coinmarketcap("stratis", "20160811", "20180324")
neo = load_price_from_coinmarketcap("neo", "20160908", "20180324")
gnt = load_price_from_coinmarketcap("golem-network-tokens", "20161118", "20180324")
kmd = load_price_from_coinmarketcap("komodo", "20170205", "20180324")
ark = load_price_from_coinmarketcap("ark", "20170523", "20180324")
qtum = load_price_from_coinmarketcap("qtum", "20170524", "20180324")


writer = pd.ExcelWriter('Competing Projects.xlsx', engine='xlsxwriter')
lsk.to_excel(writer, sheet_name="lsk")
waves.to_excel(writer, sheet_name="waves")
strat.to_excel(writer, sheet_name="strat")
neo.to_excel(writer, sheet_name="neo")
gnt.to_excel(writer, sheet_name="gnt")
kmd.to_excel(writer, sheet_name="kmd")
ark.to_excel(writer, sheet_name="ark")
qtum.to_excel(writer, sheet_name="qtum")

writer.save()

writer = pd.ExcelWriter('BTC Price.xlsx', engine='xlsxwriter')
btc.to_excel(writer, sheet_name="btc")
writer.save()