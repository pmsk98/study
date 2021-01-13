# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:05:22 2021

@author: 박명석
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

path = "C:/Users/user/Desktop/연구/시가총액100위"

file_list =os.listdir(path)

df= []

for file in file_list:
    path = "C:/Users/user/Desktop/연구/시가총액100위"
    
    df.append(pd.read_csv(path+"/"+file))


for  i in df:
    ADX=talib.ADX(i.High,i.Low,i.Close,timeperiod=14)
 
    aroondown,aroonup =talib.AROON(i.High, i.Low, timeperiod=14)
    
    AROONOSC=talib.AROONOSC(i.High,i.Low,timeperiod=14)
    
    BOP=talib.BOP(i.Open,i.High,i.Low,i.Close)
    
    CCI=talib.CCI(i.High,i.Low,i.Close,timeperiod=9)
    
    CMO=talib.CMO(i.Close,timeperiod=14)
    
    DX=talib.DX(i.High,i.Low,i.Close,timeperiod=14)
    
    MFI=talib.MFI(i.High, i.Low,i.Close, i.Volume, timeperiod=14)
    
    PPO=talib.PPO(i.Close, fastperiod=12, slowperiod=26, matype=0)
    
    ROC=talib.ROC(i.Close,timeperiod=10)
    
    RSI=talib.RSI(i.Close,timeperiod=14)
    
    slowk, slowd = talib.STOCH(i.High, i.Low, i.Close, fastk_period=12.5, slowk_period=5, slowk_matype=0, slowd_period=3, slowd_matype=0)
    
    fastk, fastd = talib.STOCHF(i.High, i.Low, i.Close, fastk_period=5, fastd_period=5.3, fastd_matype=0)
    
    ULTOSC=talib.ULTOSC(i.High,i.Low,i.Close,timeperiod1=7,timeperiod2=14,timeperiod3=28)
    
    WILLR=talib.WILLR(i.High,i.Low,i.Close,timeperiod=14)
        
    i['ADX']=ADX
    i['aroondown']=aroondown
    i['aroonup']=aroonup
    i['BOP']=BOP
    i['CCI']=CCI
    i['CMO']=CMO
    i['DX']=DX
    i['MFI']=MFI
    i['PPO']=PPO
    i['ROC']=ROC
    i['RSI']=RSI
    i['slowk']=slowk
    i['slowd']=slowd
    i['fastk']=fastk
    i['fastd']=fastd
    i['ULTOSC']=ULTOSC
    i['WILLR']=WILLR


for i in df:
    i['diff']=i.Close.diff().shift(-1).fillna(0)
    i['Label'] = None
    
for i in df:
    for e in range(len(i['diff'])):    
        if i['diff'][e] > 0:
            i['Label'][e] = '1'
        elif i['diff'][e]==0:
            i['Label'][e] ='0'
        else:        
            i['Label'][e] = '0'
            


df

for i in df:
    i= i.drop(['diff'],axis=1)




for i in range(0,75):
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)

#주가 동향 확인

from datetime import datetime
import time
import matplotlib.font_manager as fm


path='C:/Users/user/Desktop/연구/NanumBarunGothic.ttf'
fontprop =fm.FontProperties(fname=path,size=15)



time_format="%Y-%m-%d"



for i in range(0,75):
    df[i]['Date']=pd.to_datetime(df[i]['Date'],format=time_format)
    


for i in range(0,75):
    
    plt.title('{}'.format(file_list[i]),fontproperties=fontprop)
    plt.plot(df[i][['Date']],df[i][['Close']])
    plt.show()



#model train/test set 생성  
train_data=[]



test_data=[]


for i in range(0,75):
    train=df[i]['Date'].str.contains('2009|2010|2011|2012|2013|2014|2015|2016|2017|2018|2019')
    train_data.append(df[i][train])
for i in range(0,75):    
    test=df[i]['Date'].str.contains('2020')
    test_data.append(df[i][test])
    



for i in range(0,75):
    train_data[i]=train_data[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
    test_data[i]=test_data[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)
    

#x_train,y_train,x_test,y_test

x_train =[]
y_train =[]
x_test=[]
y_test=[]


for i in range(0,75):
    x_train.append(train_data[i].drop(['Label'],axis=1))
    y_train.append(train_data[i]['Label'])
    
    x_test.append(test_data[i].drop(['Label'],axis=1))
    y_test.append(test_data[i]['Label']) 
    
#모델링
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier





pred=[]
pred_decision=[]
pred_naive=[]
pred_randomforest=[]
pred_svm=[]
pred_knn=[]
pred_neural=[]
pred_voting=[]

for i in range(0,75):
    #logistic
    logistic =LogisticRegression()
    logistic.fit(x_train[i],y_train[i])
    
    pred.append(logistic.predict(x_test[i]))
    #decision tree
    dt=DecisionTreeClassifier()
    
    dt.fit(x_train[i],y_train[i])
    pred_decision.append(dt.predict(x_test[i]))
    
    #naive
    naive=GaussianNB()
    
    naive.fit(x_train[i],y_train[i])
    
    pred_naive.append(naive.predict(x_test[i]))
    
    #randomforest
    randomforest=RandomForestClassifier()
    
    randomforest.fit(x_train[i],y_train[i])
    
    pred_randomforest.append(randomforest.predict(x_test[i]))
    
    #svm
    svm=SVC()
    
    svm.fit(x_train[i],y_train[i])
    
    pred_svm.append(svm.predict(x_test[i]))
    
    #knn
    knn=KNeighborsClassifier()
    
    knn.fit(x_train[i],y_train[i])
    
    pred_knn.append(knn.predict(x_test[i]))
    #nueral
    
    nueral=MLPClassifier()
    
    nueral.fit(x_train[i],y_train[i])
    
    pred_neural.append(nueral.predict(x_test[i]))
    
    
    #voting
    
    voting=VotingClassifier(estimators=[('decison',dt),('knn',knn),('logisitc',logistic),('svm',svm),('randomforest',randomforest),
                                        ('naive',naive),('nueral',nueral)],voting='hard')
    
    voting.fit(x_train[i],y_train[i])
    
    pred_voting.append(voting.predict(x_test[i]))



        
#예측값 붙이기 19년도 데이터에 붙이기

test_2019=[]

for i in range(0,75):    
    test_19=df[i]['Date'].str.contains('2020')
    test_2019.append(df[i][test_19])
    


for i in range(0,75):
    test_2019[i]['pred']=pred[i]
    test_2019[i]['pred_decision']=pred_decision[i]
    test_2019[i]['pred_naive']=pred_naive[i]
    test_2019[i]['pred_randomforest']=pred_randomforest[i]
    test_2019[i]['pred_svm']=pred_svm[i]
    test_2019[i]['pred_knn']=pred_knn[i]
    test_2019[i]['pred_neural']=pred_neural[i]
    test_2019[i]['pred_voting']=pred_voting[i]


acc =[]
acc1=[]
acc2=[]
acc3=[]
acc4=[]
acc5=[]
acc6=[]
acc7=[]


for i in range(0,75):
    acc.append(accuracy_score(test_2019[i]['Label'],test_2019[i]['pred']))
    acc1.append(accuracy_score(test_2019[i]['Label'],test_2019[i]['pred_decision']))
    acc2.append(accuracy_score(test_2019[i]['Label'],test_2019[i]['pred_naive']))
    acc3.append(accuracy_score(test_2019[i]['Label'],test_2019[i]['pred_randomforest']))
    acc4.append(accuracy_score(test_2019[i]['Label'],test_2019[i]['pred_svm']))
    acc5.append(accuracy_score(test_2019[i]['Label'],test_2019[i]['pred_knn']))
    acc6.append(accuracy_score(test_2019[i]['Label'],test_2019[i]['pred_neural']))
    acc7.append(accuracy_score(test_2019[i]['Label'],test_2019[i]['pred_voting']))


np.mean(acc)
np.mean(acc1)
np.mean(acc2)
np.mean(acc3)
np.mean(acc4)
np.mean(acc5)
np.mean(acc6)




#pred 자료형 변경
for i in range(0,75):
    test_2019[i]['pred']=test_2019[i]['pred'].astype('float')
    test_2019[i]['pred_decision']=test_2019[i]['pred_decision'].astype('float')
    test_2019[i]['pred_naive']=test_2019[i]['pred_naive'].astype('float')
    test_2019[i]['pred_randomforest']=test_2019[i]['pred_randomforest'].astype('float')
    test_2019[i]['pred_svm']=test_2019[i]['pred_svm'].astype('float')
    test_2019[i]['pred_knn']=test_2019[i]['pred_knn'].astype('float')
    test_2019[i]['pred_neural']=test_2019[i]['pred_neural'].astype('float')
    test_2019[i]['pred_voting']=test_2019[i]['pred_voting'].astype('float')
    
    

#새로운 라벨 추가
for i in range(0,75):
    test_2019[i]['position']=None

for i in range(0,75):
    for e in range(1510,1761):
        if test_2019[i]['pred'][e]+test_2019[i]['pred'][e+1]==0:
            test_2019[i]['position'][e+1]='no action'
        elif test_2019[i]['pred'][e]+test_2019[i]['pred'][e+1]==2:
            test_2019[i]['position'][e+1]='holding'
        elif test_2019[i]['pred'][e] > test_2019[i]['pred'][e+1]:
            test_2019[i]['position'][e+1]='sell'
        else:
            test_2019[i]['position'][e+1]='buy'
            
            
#decion tree 기준
for i in range(0,75):
    for e in range(1510,1761):
        if test_2019[i]['pred_decision'][e]+test_2019[i]['pred_decision'][e+1]==0:
            test_2019[i]['position'][e+1]='no action'
        elif test_2019[i]['pred_decision'][e]+test_2019[i]['pred_decision'][e+1]==2:
            test_2019[i]['position'][e+1]='holding'
        elif test_2019[i]['pred_decision'][e] > test_2019[i]['pred_decision'][e+1]:
            test_2019[i]['position'][e+1]='sell'
        else:
            test_2019[i]['position'][e+1]='buy'

#naive
for i in range(0,75):
    for e in range(1510,1761):
        if test_2019[i]['pred_naive'][e]+test_2019[i]['pred_naive'][e+1]==0:
            test_2019[i]['position'][e+1]='no action'
        elif test_2019[i]['pred_naive'][e]+test_2019[i]['pred_naive'][e+1]==2:
            test_2019[i]['position'][e+1]='holding'
        elif test_2019[i]['pred_naive'][e] > test_2019[i]['pred_naive'][e+1]:
            test_2019[i]['position'][e+1]='sell'
        else:
            test_2019[i]['position'][e+1]='buy'

#randomforest
for i in range(0,75):
    for e in range(1510,1761):
        if test_2019[i]['pred_randomforest'][e]+test_2019[i]['pred_randomforest'][e+1]==0:
            test_2019[i]['position'][e+1]='no action'
        elif test_2019[i]['pred_randomforest'][e]+test_2019[i]['pred_randomforest'][e+1]==2:
            test_2019[i]['position'][e+1]='holding'
        elif test_2019[i]['pred_randomforest'][e] > test_2019[i]['pred_randomforest'][e+1]:
            test_2019[i]['position'][e+1]='sell'
        else:
            test_2019[i]['position'][e+1]='buy'


#svm
for i in range(0,75):
    for e in range(1510,1761):
        if test_2019[i]['pred_svm'][e]+test_2019[i]['pred_svm'][e+1]==0:
            test_2019[i]['position'][e+1]='no action'
        elif test_2019[i]['pred_svm'][e]+test_2019[i]['pred_svm'][e+1]==2:
            test_2019[i]['position'][e+1]='holding'
        elif test_2019[i]['pred_svm'][e] > test_2019[i]['pred_svm'][e+1]:
            test_2019[i]['position'][e+1]='sell'
        else:
            test_2019[i]['position'][e+1]='buy'

#knn
for i in range(0,75):
    for e in range(1510,1761):
        if test_2019[i]['pred_knn'][e]+test_2019[i]['pred_knn'][e+1]==0:
            test_2019[i]['position'][e+1]='no action'
        elif test_2019[i]['pred_knn'][e]+test_2019[i]['pred_knn'][e+1]==2:
            test_2019[i]['position'][e+1]='holding'
        elif test_2019[i]['pred_knn'][e] > test_2019[i]['pred_knn'][e+1]:
            test_2019[i]['position'][e+1]='sell'
        else:
            test_2019[i]['position'][e+1]='buy'


#nueral
for i in range(0,75):
    for e in range(1510,1761):
        if test_2019[i]['pred_neural'][e]+test_2019[i]['pred_neural'][e+1]==0:
            test_2019[i]['position'][e+1]='no action'
        elif test_2019[i]['pred_neural'][e]+test_2019[i]['pred_neural'][e+1]==2:
            test_2019[i]['position'][e+1]='holding'
        elif test_2019[i]['pred_neural'][e] > test_2019[i]['pred_neural'][e+1]:
            test_2019[i]['position'][e+1]='sell'
        else:
            test_2019[i]['position'][e+1]='buy'

#voting
for i in range(0,75):
    for e in range(1510,1761):
        if test_2019[i]['pred_voting'][e]+test_2019[i]['pred_voting'][e+1]==0:
            test_2019[i]['position'][e+1]='no action'
        elif test_2019[i]['pred_voting'][e]+test_2019[i]['pred_voting'][e+1]==2:
            test_2019[i]['position'][e+1]='holding'
        elif test_2019[i]['pred_voting'][e] > test_2019[i]['pred_voting'][e+1]:
            test_2019[i]['position'][e+1]='sell'
        else:
            test_2019[i]['position'][e+1]='buy'


#첫날 position이 holding일 경우 buy로 변경
for i in range(0,75):
    if test_2019[i]['position'][1511]=='holding':
        test_2019[i]['position'][1511]='buy'
    elif test_2019[i]['position'][1511]=='sell':
        test_2019[i]['position'][1511]='buy'
    else:
        print(i)
        
        
      
#강제 청산
for i in range(0,75):
    if test_2019[i]['position'][1761]=='holding':
        test_2019[i]['position'][1761]='sell'
    elif test_2019[i]['position'][1761]=='buy':
        test_2019[i]['position'][1761]='sell'
    elif test_2019[i]['position'][1761]=='no action':
        test_2019[i]['position'][1761]='sell'
    else:
        print(i)
        
        
#profit 생성
len(test_2019)
for i in range(0,75):
    test_2019[i]['profit']=None
    
#다음날 수정 종가를 가져오게 생성
for i in range(0,75):
    for e in range(1510,1761):
        if test_2019[i]['position'][e]=='buy':
            test_2019[i]['profit'][e]=test_2019[i]['Adjusted'][e+1]
        elif test_2019[i]['position'][e]=='sell':
            test_2019[i]['profit'][e]=test_2019[i]['Adjusted'][e+1]
        else:
            print(i)


for i in range(0,75):
    if test_2019[i]['position'][1761]=='sell':
        test_2019[i]['profit'][1761]=test_2019[i]['Adjusted'][1761]
        
####

buy_label=[]
for i in range(0,75):
    buy_position=test_2019[i]['position']=='buy'
    buy_label.append(test_2019[i][buy_position])
    
sell_label=[]
for i in range(0,75):
    sell_position=test_2019[i]['position']=='sell'
    sell_label.append(test_2019[i][sell_position])    


buy=[]
sell=[]
for i in range(0,75):
    buy.append(buy_label[i]['Adjusted'].reset_index(drop=True))
    sell.append(sell_label[i]['Adjusted'].reset_index(drop=True))
    
  
profit_2=[]    
for i in range(0,75):
    profit_2.append(sell[i]-buy[i])
  

for i in range(0,75):
    test_2019[i]['profit_2']=None
    

#profit 결측치 처리
for i in range(0,75):
    profit_2[i]=profit_2[i].dropna()
    
    
#profit_2 sell에 해당하는 행에 값 넣기
for tb, pf in zip(test_2019, profit_2):
    total_idx = tb[tb['position'] == 'sell'].index
    total_pf_idx = pf.index
    for idx, pf_idx in zip(total_idx, total_pf_idx):
        tb.loc[idx, 'profit_2'] = pf[pf_idx]
        

for i in range(0,75):
    test_2019[i]['profit_cumsum']=None
    
    
    

#profit 누적 합 
for i in range(0,75):
    for e in range(1510,1761):
        if test_2019[i]['position'][e]=='holding':
            test_2019[i]['profit_2'][e]=0
        elif test_2019[i]['position'][e]=='no action':
            test_2019[i]['profit_2'][e]=0
        elif test_2019[i]['position'][e]=='buy':
            test_2019[i]['profit_2'][e]=0
        else:
            print(i)


#새로운 청산 기준 누적합



for i in range(0,75):
    test_2019[i]['profit_cumsum2']=None    
    
    
for i in range(0,75):
    test_2019[i]['profit_cumsum']=test_2019[i]['profit_2'].cumsum()





######누적합 그래프
from datetime import datetime
import time

time_format="%Y-%m-%d"



for i in range(0,75):
    test_2019[i]['Date']=pd.to_datetime(test_2019[i]['Date'],format=time_format)
    
    
#diff 컬럼 생성(새로운 청산 기준)
for i in range(0,75):
    test_2019[i]['diff']=test_2019[i]['Adjusted'].diff()
    
for i in range(0,75):
    test_2019[i]['diff'][1510]=0


for i in range(0,75):
    test_2019[i]['profit_cumsum2']=None

for i in range(0,75):
    test_2019[i]['profit_cumsum2']=test_2019[i]['diff'].cumsum()



test_2019[0]['profit_cumsum2']
#그래프 생성
import matplotlib.pyplot as plt
import seaborn as sns
for i in range(0,75):
    plt.title('{}_voting'.format(file_list[i]))
    plt.plot(test_2019[i][['Date']],test_2019[i][['profit_cumsum']],color='blue')
    plt.plot(test_2019[i][['Date']],test_2019[i][['profit_cumsum2']],color='red')
    plt.xticks(size=10)
    plt.show()
    




#ratio 작성
for i in range(0,75):
    profit_2[i]=pd.DataFrame(profit_2[i])

#거래횟수
trade= []

for i in range(0,75):
    trade.append(len(profit_2[i]))
    
#승률


for i in range(0,75):
    profit_2[i]['average']=None

   
for i in range(0,75):
    for e in range(len(profit_2[i])):      
        if profit_2[i]['Adjusted'][e] > 0:
            profit_2[i]['average'][e]='gain'
        else:
            profit_2[i]['average'][e]='loss'
            
for i in range(0,75):
    for e in range(len(profit_2[i])):
        if profit_2[i]['Adjusted'][e] < 0:
            profit_2[i]['Adjusted'][e]=profit_2[i]['Adjusted'][e] * -1
        else:
            print(i)

win=[]
for i in range(0,75):
    win.append(profit_2[i].groupby('average').size()[0]/len(profit_2[i]))
    
#평균 수익

gain=[]

for i in range(0,75):
    gain.append(profit_2[i].groupby('average').mean())
    

real_gain=[]

for i in range(0,75):
    real_gain.append(gain[i]['Adjusted'][0])
    
#평균 손실
loss=[]

for i in range(0,75):
    loss.append(gain[i]['Adjusted'][1])
  
#payoff ratio
payoff=[]

for i in range(0,75):
    payoff.append(gain[i]['Adjusted'][0]/gain[i]['Adjusted'][1])
    
#profit factor

factor_sum=[]


for i in range(0,75):
    factor_sum.append(profit_2[i].groupby('average').sum())

factor=[]

for i in range(0,75):
    factor.append(factor_sum[i]['Adjusted'][0]/factor_sum[i]['Adjusted'][1])


#최종 결과물 파일 작성
stock_name=pd.DataFrame({'stock_ticker':file_list})

stock_name=stock_name.replace('.csv','',regex=True)

trade=pd.DataFrame({'No.trades':trade})

win=pd.DataFrame({'Win%':win})

real_gain=pd.DataFrame({'Average gain($)':real_gain})

loss=pd.DataFrame({'Average loss($)':loss})

payoff=pd.DataFrame({'Payoff ratio':payoff})

factor=pd.DataFrame({'Profit factor':factor})

result =pd.concat([stock_name,trade,win,real_gain,loss,payoff,factor],axis=1)