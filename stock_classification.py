# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:37:28 2020

@author: pmsk9
"""

#주가 classification
import pandas as pd
import  talib

apple =pd.read_csv('C:/Users/pmsk9/Desktop/연구/ticker 주가/AAPL.csv')


apple.head()

ADX=talib.ADX(apple.High,apple.Low,apple.Close,timeperiod=14)

ADXR=talib.ADXR(apple.High,apple.Low,apple.Close,timeperiod=14)

APO=talib.APO(apple.Adjusted,fastperiod=12,slowperiod=26,matype=0)

aroondown,aroonup =talib.AROON(apple.High, apple.Low, timeperiod=14)

AROONOSC=talib.AROONOSC(apple.High,apple.Low,timeperiod=14)

BOP=talib.BOP(apple.Open,apple.High,apple.Low,apple.Close)

CCI=talib.CCI(apple.High,apple.Low,apple.Close,timeperiod=14)

CMO=talib.CMO(apple.Adjusted,timeperiod=14)

DX=talib.DX(apple.High,apple.Low,apple.Close,timeperiod=14)

macd, macdsignal, macdhist = talib.MACD(apple.Adjusted, fastperiod=12, slowperiod=26, signalperiod=9)

ma_macd, ma_macdsignal, ma_macdhist = talib.MACDEXT(apple.Adjusted, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)

fix_macd,fix_macdsignal, fix_macdhist = talib.MACDFIX(apple.Adjusted, signalperiod=9)

MFI=talib.MFI(apple.High, apple.Low,apple.Close, apple.Volume, timeperiod=14)

MINUS_DI=talib.MINUS_DI(apple.High, apple.Low, apple.Close, timeperiod=14)

MINUS_DM=talib. MINUS_DM(apple.High, apple.Low, timeperiod=14)

MOM=talib.MOM(apple.Adjusted,timeperiod=10)

PLUS_DM=talib.PLUS_DM(apple.High,apple.Low,timeperiod=14)

PPO=talib.PPO(apple.Adjusted, fastperiod=12, slowperiod=26, matype=0)

ROC=talib.ROC(apple.Adjusted,timeperiod=10)

ROCP=talib.ROCP(apple.Adjusted,timeperiod=10)

ROCR=talib.ROCR(apple.Adjusted,timeperiod=10)

ROCR100=talib.ROCR100(apple.Adjusted,timeperiod=10)

RSI=talib.RSI(apple.Adjusted,timeperiod=14)

slowk, slowd = talib.STOCH(apple.High, apple.Low, apple.Close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

fastk, fastd = talib.STOCHF(apple.High, apple.Low, apple.Close, fastk_period=5, fastd_period=3, fastd_matype=0)

TRIX=talib.TRIX(apple.Adjusted,timeperiod=30)

ULTOSC=talib.ULTOSC(apple.High,apple.Low,apple.Close,timeperiod1=7,timeperiod2=14,timeperiod3=28)

WILLR=talib.WILLR(apple.High,apple.Low,apple.Close,timeperiod=14)



apple.Adjusted
apple.low

