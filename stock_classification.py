
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:37:28 2020
@author: pmsk9
"""

#주가 classification
import pandas as pd
import  talib
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

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


apple.to_csv('apple_stock_.csv')

#오르면 1 내리면 0 (전날 종가 기준)
def add_label(apple):
    idx=len(apple.columns)
    new_col=np.where(apple['Close']>=apple['Close'].shift(1),'up','down')
    apple.insert(loc=idx,column='Label',value=new_col)
    apple=apple.fillna(0)

add_label(apple)
apple.head()

apple.to_csv('apple_stock_label.csv')
#컬럼이름 변경
apple =apple.rename(columns={'Unnamed: 0':'Date'})


#train data 생성
train =apple['Date'].str.contains('2014|2015|2016|2017|2018')
train_data = apple[train]
#train data 생성
x_train=train_data.drop(['Date','Label'],axis=1)
y_train=train_data['Label']

#test data 생성
test =apple['Date'].str.contains('2019')
test_data =apple[test]

x_test=test_data.drop(['Date','Label'],axis=1)
y_test=test_data['Label']

#####모델######
def svc_param_selection(X,y,nfolds):
    svm_parameters=[
        {'kernel':['rbf'],
         'gamma':[0.00001,0.0001,0.001,0.01,0.1,1]
         ,'C':[0.01,0.1,1,10,100,1000]}]
    clf =GridSearchCV(SVC(), svm_parameters, cv=10)
    clf.fit(x_train,y_train.values.ravel())
    print(clf.best_params_)
    
    return clf

#rbf ->클래스 1만 나와서 linear로 바꿔서 진행 
clf = svc_param_selection(x_train, y_train, 10)


y_true , y_pred =y_test,clf.predict(x_test)


#정규화 모델링
clf = svc_param_selection(x_scaled, y_train, 10)


y_true , y_pred =y_test,clf.predict(x_test_scaled)


print(classification_report(y_true,y_pred))
print("accuracy:"+str(accuracy_score(y_true, y_pred)))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true,y_pred)

comparison=pd.DataFrame({'prediction':y_pred,
                         'ground_truth':y_true.values.ravel()})
comparison

#정규화 standardScaling
from sklearn.preprocessing import StandardScaler

scaler =StandardScaler()
scaler.fit(x_train)

x_scaled =scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)



clf = SVC(x_scaled, y_train, 10)


y_true , y_pred =y_test,clf.fit(x_test_scaled,y_test.values.ravel())


########최종모델
#grid search 해야함
scaler =StandardScaler()
scaler.fit(x_train)

x_train_sc=scaler.transform(x_train)
x_test_sc=scaler.transform(x_test)

ml = SVC(kernel='rbf',C=1.0,random_state=0)
ml.fit(x_train_sc,y_train)

y_pred=ml.predict(x_test_sc)
y_true=y_test.values.ravel()

print(classification_report(y_true,y_pred))
print("accuracy:"+str(accuracy_score(y_true, y_pred)))


comparison=pd.DataFrame({'prediction':y_pred,
                         'ground_truth':y_true})

confusion_matrix(y_true,y_pred)
comparison


#데이터 시각화

#linear가 rbf보다 더 accuracy 측면에서 더 좋음



def svc_param_selection(X,y,nfolds):
    svm_parameters=[
        {'kernel':['rbf'],
         'gamma':[0.00001,0.0001,0.001,0.01,0.1,1]
         ,'C':[0.01,0.1,1,10,100,1000]}]
    clf =GridSearchCV(SVC(), svm_parameters, cv=10)
    clf.fit(x_train_sc,y_train.values.ravel())
    print(clf.best_params_)
    
    return clf

#rbf ->클래스 1만 나와서 linear로 바꿔서 진행 
clf = svc_param_selection(x_train_sc, y_train, 10)


y_true , y_pred =y_test,clf.predict(x_test_sc)


print(classification_report(y_true,y_pred))
print("accuracy:"+str(accuracy_score(y_true, y_pred)))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true,y_pred)

comparison=pd.DataFrame({'prediction':y_pred,
                         'ground_truth':y_true.values.ravel()})
comparison

c_canditates=[]
c_canditates.append(clf.best_params_['C']*0.01)
c_canditates.append(clf.best_params_['C'])
c_canditates.append(clf.best_params_['C']*100)

gamma_candidates =[]
gamma_candidates.append(clf.bset_parmas_['gamma']*0.01)
gamma_candidates.append(clf.best_params_['gamma'])
gamma_candidates.append(clf.best_params_['gamma']*100)

label =[]
apple.head()
for gt in apple.Label:
    if gt=='down':
        Label.append(0)
    else:
        Label.append(1)