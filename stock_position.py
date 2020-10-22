# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 23:06:26 2020

@author: pmsk9
"""

###buy,sell,holding 라벨 만들기
import pandas as pd
import  talib
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

apple =pd.read_csv('C:/Users/pmsk9/Desktop/연구/ticker 주가/AAPL.csv')



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


apple['ADX']=ADX
apple['ADXR']=ADXR
apple['APO']=APO
apple['aroondown']=aroondown
apple['aroonup']=aroonup
apple['AROONOSC']=AROONOSC
apple['BOP']=BOP
apple['CCI']=CCI
apple['CMO']=CMO
apple['DX']=DX
apple['MACD']=macd
apple['macdsignal']=macdsignal
apple['macdhist']=macdhist
apple['ma_macd']=ma_macd
apple['ma_macdsignal']=ma_macdsignal
apple['ma_macdhist']=ma_macdhist
apple['fix_macd']=fix_macd
apple['fix_macdsignal']=fix_macdsignal
apple['fix_macdhist']=fix_macdhist
apple['MFI']=MFI
apple['MINUS_DI']=MINUS_DI
apple['MINUS_DM']=MINUS_DM
apple['MOM']=MOM
apple['PLUS_DM']=PLUS_DM
apple['PPO']=PPO
apple['ROC']=ROC
apple['ROCP']=ROCP
apple['ROCR']=ROCR
apple['ROCR100']=ROCR100
apple['RSI']=RSI
apple['slowk']=slowk
apple['slowd']=slowd
apple['fastk']=fastk
apple['fastd']=fastd
apple['TRIX']=TRIX
apple['ULTOSC']=ULTOSC
apple['WILLR']=WILLR

apple.columns


#오르면 1 내리면 0 (전날 종가 기준)
apple['diff']=apple.Adjusted.diff().shift(-1).fillna(0)
apple['Label'] = None
for i in range(len(apple)):
    if apple['diff'][i] > 0:
        apple['Label'][i] = '1'
    elif apple['diff'][i]==0:
        apple['Label'][i] ='0'
    else:        
        apple['Label'][i] = '0'

apple= apple.drop(['diff'],axis=1)



apple.to_csv('apple_stock_reallabel.csv')
#컬럼이름 변경
apple =apple.rename(columns={'Unnamed: 0':'Date'})

#train/test set
train =apple['Date'].str.contains('2014|2015|2016|2017|2018')
train_data = apple[train]
#train data 생성
x_train=train_data.drop(['Date','Label','Open','High','Low','Close','Volume','Adjusted'],axis=1)
y_train=train_data['Label']




#test data 생성
test =apple['Date'].str.contains('2019')
test_data =apple[test]

x_test=test_data.drop(['Date','Label','Open','High','Low','Close','Volume','Adjusted'],axis=1)
y_test=test_data['Label']

#knn
from sklearn.neighbors import KNeighborsClassifier

knn =KNeighborsClassifier()

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

y_true=y_test

print(classification_report(y_true,y_pred))
print("accuracy:"+str(accuracy_score(y_true, y_pred)))




#2019 라벨링 추가

apple_new =apple['Date'].str.contains('2019')
apple2 = apple[apple_new]

#예측값 추가
apple2['pred']=y_pred

#position label 추가
apple2['position']=None

#변수형 타입 변환(정수형)
apple2['pred']=apple2['pred'].astype('float')



#새로운 라벨 추가
for i in range(1510,1761):
    if apple2['pred'][i]+apple2['pred'][i+1]==0:
        apple2['position'][i+1]='no action'
    elif apple2['pred'][i]+ apple2['pred'][i+1]==2:
        apple2['position'][i+1]='holding'
    elif apple2['pred'][i] > apple2['pred'][i+1]:
        apple2['position'][i+1]='sell'
    else:
        apple2['position'][i+1]='buy'
        
#####차익 계산
#buy label index

buy_position = apple2['position']=='buy'

buy_label= apple2[buy_position]


#sell label index
sell_position =apple2['position'] =='sell'

sell_label =apple2[sell_position]


####
a=sell_label['Close'].reset_index(drop=True)
b=buy_label['Close'].reset_index(drop=True)
#매도 종가 - 매수 종가
a-b

