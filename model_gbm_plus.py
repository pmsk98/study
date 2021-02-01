# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:00:39 2021

@author: student
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
from sklearn.ensemble import GradientBoostingClassifier 

 
path = "C:/Users/student/Desktop/시가총액"


file_list =os.listdir(path)


df= []


for file in file_list:

    path = "C:/Users/student/Desktop/시가총액"

    

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

 

 

 

 

for i in range(0,73):

    df[i]=df[i].drop(['Unnamed: 0','Change'],axis=1)

 

###########modeling

 

#model train/test set 생성  

train_data=[]

test_data=[]

 

 

train_data_2017=[]

test_data_2017=[]

 

train_data_2018=[]

test_data_2018=[]

 

train_data_2019=[]

test_data_2019=[]

 

 

train_data_2020=[]

test_data_2020=[]

 

train_data_2017_2=[]

test_data_2017_2=[]

 

train_data_2018_2=[]

test_data_2018_2=[]

 

train_data_2019_2=[]

test_data_2019_2=[]

 

train_data_2020_2=[]

test_data_2020_2=[]

############2016

for i in range(0,73):

    train=None

    train=df[i]['Date'].str.contains('2009|2010|2011|2012|2013|2014|2015')

    train_data.append(df[i][train])

for i in range(0,73):

    test=None    

    test=df[i]['Date'].str.contains('2016')

    test_data.append(df[i][test])

    

 

for i in range(0,73):

    train_data[i]=train_data[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

    test_data[i]=test_data[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

    

 

###############2017_1

for i in range(0,73):

    train=None

    train=df[i]['Date'].str.contains('2009|2010|2011|2012|2013|2014|2015|2016')

    train_data_2017.append(df[i][train])

for i in range(0,73):

    test=None      

    test=df[i]['Date'].str.contains('2017')

    test_data_2017.append(df[i][test])

    

 

for i in range(0,73):

    train_data_2017[i]=train_data_2017[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

    test_data_2017[i]=test_data_2017[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

 

############2018_1

for i in range(0,73):

    train=None

    train=df[i]['Date'].str.contains('2009|2010|2011|2012|2013|2014|2015|2016|2017')

    train_data_2018.append(df[i][train])

for i in range(0,73):   

    test=None

    test=df[i]['Date'].str.contains('2018')

    test_data_2018.append(df[i][test])

    

 

for i in range(0,73):

    train_data_2018[i]=train_data_2018[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

    test_data_2018[i]=test_data_2018[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

################2019_1

for i in range(0,73):

    train=None

    train=df[i]['Date'].str.contains('2009|2010|2011|2012|2013|2014|2015|2016|2017|2018')

    train_data_2019.append(df[i][train])

for i in range(0,73):  

    test=None

    test=df[i]['Date'].str.contains('2019')

    test_data_2019.append(df[i][test])

    

 

for i in range(0,73):

    train_data_2019[i]=train_data_2019[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

    test_data_2019[i]=test_data_2019[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

################2020_1

for i in range(0,73):

    train=None

    train=df[i]['Date'].str.contains('2009|2010|2011|2012|2013|2014|2015|2016|2017|2018|2019')

    train_data_2020.append(df[i][train])

for i in range(0,73):

    test=None    

    test=df[i]['Date'].str.contains('2020')

    test_data_2020.append(df[i][test])

    

 

for i in range(0,73):

    train_data_2020[i]=train_data_2020[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

    test_data_2020[i]=test_data_2020[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

 

 

#############2017_2

for i in range(0,73):

    train=None

    train=df[i]['Date'].str.contains('2010|2011|2012|2013|2014|2015|2016')

    train_data_2017_2.append(df[i][train])

for i in range(0,73):

    test=None    

    test=df[i]['Date'].str.contains('2017')

    test_data_2017_2.append(df[i][test])

    

 

for i in range(0,73):

    train_data_2017_2[i]=train_data_2017_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

    test_data_2017_2[i]=test_data_2017_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

############2018_2

for i in range(0,73):

    train=None

    train=df[i]['Date'].str.contains('2011|2012|2013|2014|2015|2016|2017')

    train_data_2018_2.append(df[i][train])

for i in range(0,73):

    test=None    

    test=df[i]['Date'].str.contains('2018')

    test_data_2018_2.append(df[i][test])

    

 

for i in range(0,73):

    train_data_2018_2[i]=train_data_2018_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

    test_data_2018_2[i]=test_data_2018_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

 

#############2019_2

for i in range(0,73):

    train=None

    train=df[i]['Date'].str.contains('2012|2013|2014|2015|2016|2017|2018')

    train_data_2019_2.append(df[i][train])

for i in range(0,73):

    test=None    

    test=df[i]['Date'].str.contains('2019')

    test_data_2019_2.append(df[i][test])

    

 

for i in range(0,73):

    train_data_2019_2[i]=train_data_2019_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

    test_data_2019_2[i]=test_data_2019_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

 

###############2020_2

for i in range(0,73):

    train=None

    train=df[i]['Date'].str.contains('2013|2014|2015|2016|2017|2018|2019')

    train_data_2020_2.append(df[i][train])

for i in range(0,73):  

    test=None

    test=df[i]['Date'].str.contains('2020')

    test_data_2020_2.append(df[i][test])

    

 

for i in range(0,73):

    train_data_2020_2[i]=train_data_2020_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

    test_data_2020_2[i]=test_data_2020_2[i].drop(['Date','Open','High','Low','Close','Volume','diff'],axis=1)

 

 

 

 

 

#x_train,y_train,x_test,y_test

 

x_train =[]

y_train =[]

x_test=[]

y_test=[]

 

x_train_1=[]

y_train_1=[]

x_test_1=[]

y_test_1=[]

 

 

x_train_2=[]

y_train_2=[]

x_test_2=[]

y_test_2=[]

 

 

x_train_3=[]

y_train_3=[]

x_test_3=[]

y_test_3=[]

 

x_train_4=[]

y_train_4=[]

x_test_4=[]

y_test_4=[]

 

x_train_5=[]

y_train_5=[]

x_test_5=[]

y_test_5=[]

 

 

x_train_6=[]

y_train_6=[]

x_test_6=[]

y_test_6=[]

 

 

x_train_7=[]

y_train_7=[]

x_test_7=[]

y_test_7=[]

 

 

x_train_8=[]

y_train_8=[]

x_test_8=[]

y_test_8=[]

 

#######2016

for i in range(0,73):

    x_train.append(train_data[i].drop(['Label'],axis=1))

    y_train.append(train_data[i]['Label'])

    

    x_test.append(test_data[i].drop(['Label'],axis=1))

    y_test.append(test_data[i]['Label']) 

########2017_1

for i in range(0,73):

    x_train_1.append(train_data_2017[i].drop(['Label'],axis=1))

    y_train_1.append(train_data_2017[i]['Label'])

    

    x_test_1.append(test_data_2017[i].drop(['Label'],axis=1))

    y_test_1.append(test_data_2017[i]['Label']) 

 

#########2018_1

for i in range(0,73):

    x_train_2.append(train_data_2018[i].drop(['Label'],axis=1))

    y_train_2.append(train_data_2018[i]['Label'])

    

    x_test_2.append(test_data_2018[i].drop(['Label'],axis=1))

    y_test_2.append(test_data_2018[i]['Label']) 

 

##########2019_1

for i in range(0,73):

    x_train_3.append(train_data_2019[i].drop(['Label'],axis=1))

    y_train_3.append(train_data_2019[i]['Label'])

    

    x_test_3.append(test_data_2019[i].drop(['Label'],axis=1))

    y_test_3.append(test_data_2019[i]['Label']) 

 

#############2020_1

for i in range(0,73):

    x_train_4.append(train_data_2020[i].drop(['Label'],axis=1))

    y_train_4.append(train_data_2020[i]['Label'])

    

    x_test_4.append(test_data_2020[i].drop(['Label'],axis=1))

    y_test_4.append(test_data_2020[i]['Label']) 

    

##############2017_@

for i in range(0,73):

    x_train_5.append(train_data_2017_2[i].drop(['Label'],axis=1))

    y_train_5.append(train_data_2017_2[i]['Label'])

    

    x_test_5.append(test_data_2017_2[i].drop(['Label'],axis=1))

    y_test_5.append(test_data_2017_2[i]['Label'])

 

 

############2018_2

for i in range(0,73):

    x_train_6.append(train_data_2018_2[i].drop(['Label'],axis=1))

    y_train_6.append(train_data_2018_2[i]['Label'])

    

    x_test_6.append(test_data_2018_2[i].drop(['Label'],axis=1))

    y_test_6.append(test_data_2018_2[i]['Label']) 

 

############2019_2

for i in range(0,73):

    x_train_7.append(train_data_2019_2[i].drop(['Label'],axis=1))

    y_train_7.append(train_data_2019_2[i]['Label'])

    

    x_test_7.append(test_data_2019_2[i].drop(['Label'],axis=1))

    y_test_7.append(test_data_2019_2[i]['Label']) 

 

############2020_2

for i in range(0,73):

    x_train_8.append(train_data_2020_2[i].drop(['Label'],axis=1))

    y_train_8.append(train_data_2020_2[i]['Label'])

    

    x_test_8.append(test_data_2020_2[i].drop(['Label'],axis=1))

    y_test_8.append(test_data_2020_2[i]['Label']) 

 

 

    

#모델링

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import VotingClassifier

 

 

 

 



pred=[[] for _ in range(9)]

pred_decision=[[] for _ in range(9)]

pred_naive=[[] for _ in range(9)]

pred_randomforest=[[] for _ in range(9)]

pred_svm=[[] for _ in range(9)]

pred_knn=[[] for _ in range(9)]

pred_neural=[[] for _ in range(9)]

pred_voting=[[] for _ in range(9)]

pred_gbm=[[] for _ in range(9)]

 


 

for i in range(0,73):

    #logistic

    logistic =LogisticRegression()

    logistic.fit(x_train[i],y_train[i])

    

    pred[0].append(logistic.predict(x_test[i]))

    #2017

    logistic.fit(x_train_1[i],y_train_1[i])

    

    pred[1].append(logistic.predict(x_test_1[i]))

    #2018

    logistic.fit(x_train_2[i],y_train_2[i])

    

    pred[2].append(logistic.predict(x_test_2[i]))

    #2019

    logistic.fit(x_train_3[i],y_train_3[i])

    

    pred[3].append(logistic.predict(x_test_3[i]))

    #2020

    logistic.fit(x_train_4[i],y_train_4[i])

    

    pred[4].append(logistic.predict(x_test_4[i]))

    

    #2017_2

    logistic.fit(x_train_5[i],y_train_5[i])

    

    pred[5].append(logistic.predict(x_test_5[i]))

    #2018_2

    logistic.fit(x_train_6[i],y_train_6[i])

    

    pred[6].append(logistic.predict(x_test_6[i]))

    #2019_@

    logistic.fit(x_train_7[i],y_train_7[i])

    

    pred[7].append(logistic.predict(x_test_7[i]))

    #2020_2

    logistic.fit(x_train_8[i],y_train_8[i])

    

    pred[8].append(logistic.predict(x_test_8[i]))

    

    

    ##############decision tree

    dt=DecisionTreeClassifier()

    

    dt.fit(x_train[i],y_train[i])

    pred_decision[0].append(dt.predict(x_test[i]))

    

    #2017

    dt.fit(x_train_1[i],y_train_1[i])

    

    pred_decision[1].append(dt.predict(x_test_1[i]))

    #2018

    dt.fit(x_train_2[i],y_train_2[i])

    

    pred_decision[2].append(dt.predict(x_test_2[i]))

    #2019

    dt.fit(x_train_3[i],y_train_3[i])

    

    pred_decision[3].append(dt.predict(x_test_3[i]))

    #2020

    dt.fit(x_train_4[i],y_train_4[i])

    

    pred_decision[4].append(dt.predict(x_test_4[i]))

    

    #2017_2

    dt.fit(x_train_5[i],y_train_5[i])

    

    pred_decision[5].append(dt.predict(x_test_5[i]))

    #2018_2

    dt.fit(x_train_6[i],y_train_6[i])

    

    pred_decision[6].append(dt.predict(x_test_6[i]))

    #2019_@

    dt.fit(x_train_7[i],y_train_7[i])

    

    pred_decision[7].append(dt.predict(x_test_7[i]))

    #2020_2

    dt.fit(x_train_8[i],y_train_8[i])

    

    pred_decision[8].append(dt.predict(x_test_8[i]))

    

    

    ##############naive

    naive=GaussianNB()

    

    naive.fit(x_train[i],y_train[i])

    

    pred_naive[0].append(naive.predict(x_test[i]))

    

    #2017

    naive.fit(x_train_1[i],y_train_1[i])

    

    pred_naive[1].append(naive.predict(x_test_1[i]))

    #2018

    naive.fit(x_train_2[i],y_train_2[i])

    

    pred_naive[2].append(naive.predict(x_test_2[i]))

    #2019

    naive.fit(x_train_3[i],y_train_3[i])

    

    pred_naive[3].append(naive.predict(x_test_3[i]))

    #2020

    naive.fit(x_train_4[i],y_train_4[i])

    

    pred_naive[4].append(naive.predict(x_test_4[i]))

    

    #2017_2

    naive.fit(x_train_5[i],y_train_5[i])

    

    pred_naive[5].append(naive.predict(x_test_5[i]))

    #2018_2

    naive.fit(x_train_6[i],y_train_6[i])

    

    pred_naive[6].append(naive.predict(x_test_6[i]))

    #2019_@

    naive.fit(x_train_7[i],y_train_7[i])

    

    pred_naive[7].append(naive.predict(x_test_7[i]))

    #2020_2

    naive.fit(x_train_8[i],y_train_8[i])

    

    pred_naive[8].append(naive.predict(x_test_8[i]))

    

    

    #############randomforest

    randomforest=RandomForestClassifier()

    

    randomforest.fit(x_train[i],y_train[i])

    

    pred_randomforest[0].append(randomforest.predict(x_test[i]))

    

    #2017

    randomforest.fit(x_train_1[i],y_train_1[i])

    

    pred_randomforest[1].append(randomforest.predict(x_test_1[i]))

    #2018

    randomforest.fit(x_train_2[i],y_train_2[i])

    

    pred_randomforest[2].append(randomforest.predict(x_test_2[i]))

    #2019

    randomforest.fit(x_train_3[i],y_train_3[i])

    

    pred_randomforest[3].append(randomforest.predict(x_test_3[i]))

    #2020

    randomforest.fit(x_train_4[i],y_train_4[i])

    

    pred_randomforest[4].append(randomforest.predict(x_test_4[i]))

    

    #2017_2

    randomforest.fit(x_train_5[i],y_train_5[i])

    

    pred_randomforest[5].append(randomforest.predict(x_test_5[i]))

    #2018_2

    randomforest.fit(x_train_6[i],y_train_6[i])

    

    pred_randomforest[6].append(randomforest.predict(x_test_6[i]))

    #2019_@

    randomforest.fit(x_train_7[i],y_train_7[i])

    

    pred_randomforest[7].append(randomforest.predict(x_test_7[i]))

    #2020_2

    randomforest.fit(x_train_8[i],y_train_8[i])

    

    pred_randomforest[8].append(randomforest.predict(x_test_8[i]))

    

    

    ###############svm

    svm=SVC()

    

    svm.fit(x_train[i],y_train[i])

    

    pred_svm[0].append(svm.predict(x_test[i]))

    

    

    #2017

    svm.fit(x_train_1[i],y_train_1[i])

    

    pred_svm[1].append(svm.predict(x_test_1[i]))

    #2018

    svm.fit(x_train_2[i],y_train_2[i])

    

    pred_svm[2].append(svm.predict(x_test_2[i]))

    #2019

    svm.fit(x_train_3[i],y_train_3[i])

    

    pred_svm[3].append(svm.predict(x_test_3[i]))

    #2020

    svm.fit(x_train_4[i],y_train_4[i])

    

    pred_svm[4].append(svm.predict(x_test_4[i]))

    

    #2017_2

    svm.fit(x_train_5[i],y_train_5[i])

    

    pred_svm[5].append(svm.predict(x_test_5[i]))

    #2018_2

    svm.fit(x_train_6[i],y_train_6[i])

    

    pred_svm[6].append(svm.predict(x_test_6[i]))

    #2019_@

    svm.fit(x_train_7[i],y_train_7[i])

    

    pred_svm[7].append(svm.predict(x_test_7[i]))

    #2020_2

    svm.fit(x_train_8[i],y_train_8[i])

    

    pred_svm[8].append(svm.predict(x_test_8[i]))

    

    

    ###############knn

    knn=KNeighborsClassifier()

    

    knn.fit(x_train[i],y_train[i])

    

    pred_knn[0].append(knn.predict(x_test[i]))

    

    

    #2017

    knn.fit(x_train_1[i],y_train_1[i])

    

    pred_knn[1].append(knn.predict(x_test_1[i]))

    #2018

    knn.fit(x_train_2[i],y_train_2[i])

    

    pred_knn[2].append(knn.predict(x_test_2[i]))

    #2019

    knn.fit(x_train_3[i],y_train_3[i])

    

    pred_knn[3].append(knn.predict(x_test_3[i]))

    #2020

    knn.fit(x_train_4[i],y_train_4[i])

    

    pred_knn[4].append(knn.predict(x_test_4[i]))

    

    #2017_2

    knn.fit(x_train_5[i],y_train_5[i])

    

    pred_knn[5].append(knn.predict(x_test_5[i]))

    #2018_2

    knn.fit(x_train_6[i],y_train_6[i])

    

    pred_knn[6].append(knn.predict(x_test_6[i]))

    #2019_@

    knn.fit(x_train_7[i],y_train_7[i])

    

    pred_knn[7].append(knn.predict(x_test_7[i]))

    #2020_2

    knn.fit(x_train_8[i],y_train_8[i])

    

    pred_knn[8].append(knn.predict(x_test_8[i]))

    

    ###############nueral

    

    nueral=MLPClassifier()

    

    nueral.fit(x_train[i],y_train[i])

    

    pred_neural[0].append(nueral.predict(x_test[i]))

    

    

    #2017

    nueral.fit(x_train_1[i],y_train_1[i])

    

    pred_neural[1].append(nueral.predict(x_test_1[i]))

    #2018

    nueral.fit(x_train_2[i],y_train_2[i])

    

    pred_neural[2].append(nueral.predict(x_test_2[i]))

    #2019

    nueral.fit(x_train_3[i],y_train_3[i])

    

    pred_neural[3].append(nueral.predict(x_test_3[i]))

    #2020

    nueral.fit(x_train_4[i],y_train_4[i])

    

    pred_neural[4].append(nueral.predict(x_test_4[i]))

    

    #2017_2

    nueral.fit(x_train_5[i],y_train_5[i])

    

    pred_neural[5].append(nueral.predict(x_test_5[i]))

    #2018_2

    nueral.fit(x_train_6[i],y_train_6[i])

    

    pred_neural[6].append(nueral.predict(x_test_6[i]))

    #2019_@

    nueral.fit(x_train_7[i],y_train_7[i])

    

    pred_neural[7].append(nueral.predict(x_test_7[i]))

    #2020_2

    nueral.fit(x_train_8[i],y_train_8[i])

    

    pred_neural[8].append(nueral.predict(x_test_8[i]))

    

    

    

    ###########voting

    

    voting=VotingClassifier(estimators=[('decison',dt),('knn',knn),('logisitc',logistic),('svm',svm),

                                        ('naive',naive),('nueral',nueral)],voting='hard')

    

    voting.fit(x_train[i],y_train[i])

    

    pred_voting[0].append(voting.predict(x_test[i]))

    

    

    #2017

    voting.fit(x_train_1[i],y_train_1[i])

    

    pred_voting[1].append(voting.predict(x_test_1[i]))

    #2018

    voting.fit(x_train_2[i],y_train_2[i])

    

    pred_voting[2].append(voting.predict(x_test_2[i]))

    #2019

    voting.fit(x_train_3[i],y_train_3[i])

    

    pred_voting[3].append(voting.predict(x_test_3[i]))

    #2020

    voting.fit(x_train_4[i],y_train_4[i])

    

    pred_voting[4].append(voting.predict(x_test_4[i]))

    

    #2017_2

    voting.fit(x_train_5[i],y_train_5[i])

    

    pred_voting[5].append(voting.predict(x_test_5[i]))

    #2018_2

    voting.fit(x_train_6[i],y_train_6[i])

    

    pred_voting[6].append(voting.predict(x_test_6[i]))

    #2019_@

    voting.fit(x_train_7[i],y_train_7[i])

    

    pred_voting[7].append(voting.predict(x_test_7[i]))

    #2020_2

    voting.fit(x_train_8[i],y_train_8[i])

    

    pred_voting[8].append(voting.predict(x_test_8[i]))

    

    

    

    #gbm

    gbm= GradientBoostingClassifier(random_state=0)

    

    gbm.fit(x_train[i],y_train[i])

    

    pred_gbm[0].append(gbm.predict(x_test[i]))

    

    

    #2017

    gbm.fit(x_train_1[i],y_train_1[i])

    

    pred_gbm[1].append(gbm.predict(x_test_1[i]))

    #2018

    gbm.fit(x_train_2[i],y_train_2[i])

    

    pred_gbm[2].append(gbm.predict(x_test_2[i]))

    #2019

    gbm.fit(x_train_3[i],y_train_3[i])

    

    pred_gbm[3].append(gbm.predict(x_test_3[i]))

    #2020

    gbm.fit(x_train_4[i],y_train_4[i])

    

    pred_gbm[4].append(gbm.predict(x_test_4[i]))

    

    #2017_2

    gbm.fit(x_train_5[i],y_train_5[i])

    

    pred_gbm[5].append(gbm.predict(x_test_5[i]))

    #2018_2

    gbm.fit(x_train_6[i],y_train_6[i])

    

    pred_gbm[6].append(gbm.predict(x_test_6[i]))

    #2019_@

    gbm.fit(x_train_7[i],y_train_7[i])

    

    pred_gbm[7].append(gbm.predict(x_test_7[i]))

    #2020_2

    gbm.fit(x_train_8[i],y_train_8[i])

    

    pred_gbm[8].append(gbm.predict(x_test_8[i]))
