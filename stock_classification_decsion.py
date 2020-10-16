# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 16:55:04 2020

@author: pmsk9
"""

#주가 decsion tree
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


#시간
train_period =apple['Date'].str.contains('2014|2015|2016|2017|2018|2019')
train_period=apple[train_period]

test_period=apple['Date'].str.contains('2019')
test_period=apple[test_period]

#모멘텀 지표 확인
mom_period=apple

from datetime import datetime
import time

time_format="%Y-%m-%d"
train_period['Date']=pd.to_datetime(train_period['Date'],format=time_format)
test_period['Date']=pd.to_datetime(test_period['Date'],format=time_format)

mom_period['Date']=pd.to_datetime(mom_period['Date'],format=time_format)

train_period=train_period[['Date','Close']]
test_period=test_period[['Date','Close']]


#train/test 시각화
plt.title('train/test')
plt.plot(train_period[['Date']],train_period[['Close']],color='blue')
plt.plot(test_period[['Date']],test_period[['Close']],color='red')
plt.axvline(x=datetime(2018,12,31),color='r',linestyle='--',linewidth=3)
plt.show()


#모멘텀지표 시각화
for i in train_period.columns[7:44]:
    plt.title(i)
    plt.plot(train_period[['Date']],train_period[[i]],color='blue')
    plt.plot(test_period[['Date']],test_period[[i]],color='red')
    plt.axvline(x=datetime(2018,12,31),color='r',linestyle='--',linewidth=3)
    plt.show()
    
#모멘텀 지표 시각화 2013
for i in mom_period.columns[7:44]:
    plt.title(i)
    plt.plot(mom_period[['Date']],mom_period[[i]],color='blue')
    plt.show()
    
    
train_period.columns[7:44]

#label 분포 확인
#train

train_data['Label'].value_counts()

train_data['Label'].value_counts().plot(kind='bar')
plt.title('train data label')
plt.show()
#test
test_data['Label'].value_counts()

test_data['Label'].value_counts().plot(kind='bar')
plt.title('test data label')
plt.show()




#train data 생성
train
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

from sklearn.tree import DecisionTreeClassifier

dt= DecisionTreeClassifier()

dt.fit(x_train,y_train)

y_pred =dt.predict(x_test)

y_true=y_test


print(classification_report(y_true,y_pred))
print("accuracy:"+str(accuracy_score(y_true, y_pred)))


from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_true,y_pred)

#confusion matrix  시각화
plt.figure(figsize=(100,100))
sns.heatmap(confusion,annot=True,fmt='d')

#예측값과 실제값 비교
comparison=pd.DataFrame({'prediction':y_pred,
                         'ground_truth':y_true.values.ravel()})
comparison

#모델링 svm
svm_model =SVC(kernel='rbf')

svm_model.fit(x_train,y_train)

y_pred =svm_model.predict(x_test)
y_true=y_test


print(classification_report(y_true,y_pred))
print("accuracy:"+str(accuracy_score(y_true, y_pred)))


from sklearn.metrics import confusion_matrix

import seaborn as sns
confusion=confusion_matrix(y_true,y_pred)

#confusion matrix  시각화
plt.figure(figsize=(100,100))
sns.heatmap(confusion,annot=True,fmt='d')


#로지스틱 회귀분석
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import statsmodels.api as sm

#default
logistic =LogisticRegression()

logistic.fit(x_train,y_train)

y_pred=logistic.predict(x_test)

y_true=y_test

print(classification_report(y_true,y_pred))
print("accuracy:"+str(accuracy_score(y_true, y_pred)))

#gridsearch
from sklearn.model_selection import GridSearchCV

gird ={"C":np.logspace(-3,3,7),"penalty":["l1","l2"]} #l1:lasso l2:ridge
logistic=LogisticRegression()
log_cv =GridSearchCV(logistic,gird,cv=10)

log_cv.fit(x_train,y_train)

#파리미터 조정 결과
print("tuned hpyerparameters :(best parameters) ",log_cv.best_params_)
#train set 성능
print("accuracy :",log_cv.best_score_)

y_pred=log_cv.predict(x_test)

y_true=y_test

print(classification_report(y_true,y_pred))
print("accuracy:"+str(accuracy_score(y_true, y_pred)))

