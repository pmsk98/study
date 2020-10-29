# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:39:17 2020

@author: pmsk9
"""

import glob
import os
import pandas as pd
import  talib
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

path = "C:/Users/pmsk9/Desktop/연구/ticker 주가"

file_list =os.listdir(path)

df= []
for file in file_list:
    path = "C:/Users/pmsk9/Desktop/연구/ticker 주가"
    
    df.append(pd.read_csv(path+"/"+file))
    

for  i in df:
    ADX=talib.ADX(i.High,i.Low,i.Close,timeperiod=14)

    ADXR=talib.ADXR(i.High,i.Low,i.Close,timeperiod=14)
    
    APO=talib.APO(apple.Adjusted,fastperiod=10,slowperiod=20,matype=0)

    aroondown,aroonup =talib.AROON(i.High, i.Low, timeperiod=14)
    
    AROONOSC=talib.AROONOSC(i.High,i.Low,timeperiod=14)
    
    BOP=talib.BOP(i.Open,i.High,i.Low,i.Close)
    
    CCI=talib.CCI(i.High,i.Low,i.Close,timeperiod=9)
    
    CMO=talib.CMO(i.Adjusted,timeperiod=14)
    
    DX=talib.DX(i.High,i.Low,i.Close,timeperiod=14)
    
    macd, macdsignal, macdhist = talib.MACD(i.Adjusted, fastperiod=12, slowperiod=26, signalperiod=9)
    
    ma_macd, ma_macdsignal, ma_macdhist = talib.MACDEXT(i.Adjusted, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    
    fix_macd,fix_macdsignal, fix_macdhist = talib.MACDFIX(i.Adjusted, signalperiod=9)
    
    MFI=talib.MFI(i.High, i.Low,i.Close, i.Volume, timeperiod=14)
    
    MINUS_DI=talib.MINUS_DI(i.High, i.Low, i.Close, timeperiod=14)
    
    MINUS_DM=talib. MINUS_DM(i.High, i.Low, timeperiod=14)
    
    MOM=talib.MOM(i.Adjusted,timeperiod=10)
    
    PLUS_DM=talib.PLUS_DM(i.High,i.Low,timeperiod=14)
    
    PPO=talib.PPO(i.Adjusted, fastperiod=12, slowperiod=26, matype=0)
    
    ROC=talib.ROC(i.Adjusted,timeperiod=10)
    
    ROCP=talib.ROCP(i.Adjusted,timeperiod=10)
    
    ROCR=talib.ROCR(i.Adjusted,timeperiod=10)
    
    ROCR100=talib.ROCR100(i.Adjusted,timeperiod=10)
    
    RSI=talib.RSI(i.Adjusted,timeperiod=14)
    
    slowk, slowd = talib.STOCH(i.High, i.Low, i.Close, fastk_period=12.5, slowk_period=5, slowk_matype=0, slowd_period=3, slowd_matype=0)
    
    fastk, fastd = talib.STOCHF(i.High, i.Low, i.Close, fastk_period=5, fastd_period=5.3, fastd_matype=0)
    
    TRIX=talib.TRIX(i.Adjusted,timeperiod=12)
    
    ULTOSC=talib.ULTOSC(i.High,i.Low,i.Close,timeperiod1=7,timeperiod2=14,timeperiod3=28)
    
    WILLR=talib.WILLR(i.High,i.Low,i.Close,timeperiod=14)
        
    i['ADX']=ADX
    i['ADXR']=ADXR
    i['APO']=APO
    i['aroondown']=aroondown
    i['aroonup']=aroonup
    i['BOP']=BOP
    i['CCI']=CCI
    i['CMO']=CMO
    i['DX']=DX
    i['MACD']=macd
    i['macdsignal']=macdsignal
    i['macdhist']=macdhist
    i['ma_macd']=ma_macd
    i['ma_macdsignal']=ma_macdsignal
    i['ma_macdhist']=ma_macdhist
    i['fix_macd']=fix_macd
    i['fix_macdsignal']=fix_macdsignal
    i['fix_macdhist']=fix_macdhist
    i['MFI']=MFI
    i['MINUS_DI']=MINUS_DI
    i['MINUS_DM']=MINUS_DM
    i['MOM]=MOM
    i['PLUS_DM']=PLUS_DM
    i['PPO]=PPO
    i['ROC']=ROC
    i['ROCP']=ROCP
    i['ROCR']=ROCR
    i['ROCR100']=ROCR100
    i['RSI']=RSI
    i['slowk']=slowk
    i['slowd']=slowd
    i['fastk']=fastk
    i['fastd']=fastd
    i['TRIX']=TRIX
    i['ULTOSC']=ULTOSC
    i['WILLR']=WILLR


for i in df:
    i['diff']=i.Adjusted.diff().shift(-1).fillna(0)
    i['Label'] = None
    
for i in df:
    for e in range(0,1929):    
        if i['diff'][e] > 0:
            i['Label'][e] = '1'
        elif i['diff'][e]==0:
            i['Label'][e] ='0'
        else:        
            i['Label'][e] = '0'
            
len(df)

for i in df:
    i= i.drop(['diff'],axis=1)
    
    
for i in range(len(df)):
    df[i].to_csv('{}'.format(file_list[i]))
    
    