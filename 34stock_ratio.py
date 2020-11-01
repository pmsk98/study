# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:37:43 2020

@author: user
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

path = "C:/Users/user/Desktop/연구/ticker주가"

file_list =os.listdir(path)

df= []
for file in file_list:
    path = "C:/Users/user/Desktop/연구/ticker주가"
    
    df.append(pd.read_csv(path+"/"+file))


for i in range(0,34):
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    df[i]=df[i].rename(columns={'Unnamed: 0.1':'Date'})
    

for i in range(0,34):
    df[i]=df[i].rename(columns={'Unnamed: 0.1':'Date'})
 
    
train_data=[]

test_data=[]

for i in range(0,34):
    train=df[i]['Date'].str.contains('2014|2015|2016|2017|2018')
    train_data.append(df[i][train])
for i in range(0,34):    
    test=df[i]['Date'].str.contains('2019')
    test_data.append(df[i][test])
    
len(df)


for i in range(0,34):
    train_data[i]=train_data[i].drop(['Date','Open','High','Low','Close','Volume','Adjusted','diff'],axis=1)
    test_data[i]=test_data[i].drop(['Date','Open','High','Low','Close','Volume','Adjusted','diff'],axis=1)
    

#x_train,y_train,x_test,y_test

x_train =[]
y_train =[]
x_test=[]
y_test=[]


for i in range(0,34):
    x_train.append(train_data[i].drop(['Label'],axis=1))
    y_train.append(train_data[i]['Label'])
    
    x_test.append(test_data[i].drop(['Label'],axis=1))
    y_test.append(test_data[i]['Label']) 
    
#모델링
from sklearn.linear_model import LogisticRegression
pred =[]
for i in range(0,34):
    logistic =LogisticRegression()
    logistic.fit(x_train[i],y_train[i])
    
    pred.append(logistic.predict(x_test[i]))
        
#예측값 붙이기 19년도 데이터에 붙이기
test_2019=[]

for i in range(0,34):    
    test_19=df[i]['Date'].str.contains('2019')
    test_2019.append(df[i][test_19])
    


for i in range(0,34):
    test_2019[i]['pred']=pred[i]



for i in range(0,34):
    test_2019[i]['pred']=test_2019[i]['pred'].astype('float')

#새로운 라벨 추가
for i in range(0,34):
    test_2019[i]['position']=None

for i in range(0,34):
    for e in range(1510,1761):
        if test_2019[i]['pred'][e]+test_2019[i]['pred'][e+1]==0:
            test_2019[i]['position'][e+1]='no action'
        elif test_2019[i]['pred'][e]+test_2019[i]['pred'][e+1]==2:
            test_2019[i]['position'][e+1]='holding'
        elif test_2019[i]['pred'][e] > test_2019[i]['pred'][e+1]:
            test_2019[i]['position'][e+1]='sell'
        else:
            test_2019[i]['position'][e+1]='buy'


#첫날 position이 holding일 경우 buy로 변경
for i in range(0,34):
    if test_2019[i]['position'][1511]=='holding':
        test_2019[i]['position'][1511]='buy'
    else:
        print(i)
        
#강제 청산
for i in range(0,34):
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
for i in range(0,34):
    test_2019[i]['profit']=None
    
#다음날 수정 종가를 가져오게 생성
for i in range(0,34):
    for e in range(1510,1761):
        if test_2019[i]['position'][e]=='buy':
            test_2019[i]['profit'][e]=test_2019[i]['Adjusted'][e+1]
        elif test_2019[i]['position'][e]=='sell':
            test_2019[i]['profit'][e]=test_2019[i]['Adjusted'][e+1]
        else:
            print(i)


for i in range(0,34):
    if test_2019[i]['position'][1761]=='sell':
        test_2019[i]['profit'][1761]=test_2019[i]['Adjusted'][1761]
        
####

buy_label=[]
for i in range(0,34):
    buy_position=test_2019[i]['position']=='buy'
    buy_label.append(test_2019[i][buy_position])
    
sell_label=[]
for i in range(0,34):
    sell_position=test_2019[i]['position']=='sell'
    sell_label.append(test_2019[i][sell_position])    


buy=[]
sell=[]
for i in range(0,34):
    buy.append(buy_label[i]['Adjusted'].reset_index(drop=True))
    sell.append(sell_label[i]['Adjusted'].reset_index(drop=True))
    
  
profit_2=[]    
for i in range(0,34):
    profit_2.append(sell[i]-buy[i])
  
for i in range(0,34):
    test_2019[i]['profit_2']=None
    

#profit 결측치 처리
for i in range(0,34):
    profit_2[i]=profit_2[i].dropna()
    
    
#profit_2 sell에 해당하는 행에 값 넣기
for tb, pf in zip(test_2019, profit_2):
    total_idx = tb[tb['position'] == 'sell'].index
    total_pf_idx = pf.index
    for idx, pf_idx in zip(total_idx, total_pf_idx):
        tb.loc[idx, 'profit_2'] = pf[pf_idx]
        

for i in range(0,34):
    test_2019[i]['profit_cumsum']=None

#profit 누적 합 
for i in range(0,34):
    for e in range(1510,1761):
        if test_2019[i]['position'][e]=='holding':
            test_2019[i]['profit_2'][e]=0
        elif test_2019[i]['position'][e]=='no action':
            test_2019[i]['profit_2'][e]=0
        elif test_2019[i]['position'][e]=='buy':
            test_2019[i]['profit_2'][e]=0
        else:
            print(i)

for i in range(0,34):
    test_2019[i]['profit_cumsum']=test_2019[i]['profit_2'].cumsum()
    

######누적합 그래프
from datetime import datetime
import time

time_format="%Y-%m-%d"



for i in range(0,34):
    test_2019[i]['Date']=pd.to_datetime(test_2019[i]['Date'],format=time_format)


for i in range(0,34):
    plt.title('{}'.format(file_list[i]))
    plt.plot(test_2019[i][['Date']],test_2019[i][['profit_cumsum']],color='blue')
    plt.show()
    

#결과파일 저장
for i in range(0,34):
    test_2019[i].to_csv('{}'.format(file_list[i]))

#ratio 작성