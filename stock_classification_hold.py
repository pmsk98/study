# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:51:43 2020

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

apple=apple.iloc[0:365]
apple
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


#오르면 1 내리면 0 (전날 종가 기준)
apple['diff']=apple.Adjusted.diff().shift(-1).fillna(0)
apple['Label'] = None
for i in range(len(apple)):
    if apple['diff'][i] > 0:
        apple['Label'][i] = '1'
    elif apple['diff'][i]==0:
        apple['Label'][i] ='hold'
    else:        
        apple['Label'][i] = '0'

apple= apple.drop(['diff'],axis=1)


#hold label diff==0
hold=apple['Label'].str.contains('hold')
hold_label = apple[hold]

#hold label은 삭제
apple=apple[apple['Label'] != 'hold']


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

#train/test 개수
x_train.shape
y_train.shape

x_test.shape
y_test.shape

##################Class 분포 확인
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(100,100))
apple
sns.heatmap(data=apple.corr(),annot=True,fmt='.2f',linewidths=.5,cmap='Blues')

apple_visual =train_period.drop(['Date','Open','High','Low','Close','Adjusted'],axis=1)
for cnt, col in enumerate(apple_visual):
    try:
        plt.figure(figsize=(10, 5))
        sns.distplot(apple_visual[col][apple_visual['Label']=='0'])
        sns.distplot(apple_visual[col][apple_visual['Label']=='1'])
        plt.legend(['down','up'], loc='best')
        plt.title('histogram of features(train) '+str(col))
        plt.show()

        if cnt >= 40: # 6개 칼럼까지만 출력
            break

    except Exception as e:
        pass
    
apple_visual =test_period.drop(['Date','Open','High','Low','Close','Adjusted'],axis=1)
for cnt, col in enumerate(apple_visual):
    try:
        plt.figure(figsize=(10, 5))
        sns.distplot(apple_visual[col][apple_visual['Label']=='0'])
        sns.distplot(apple_visual[col][apple_visual['Label']=='1'])
        plt.legend(['down','up'], loc='best')
        plt.title('histogram of features (test)'+str(col))
        plt.show()

        if cnt >= 40: # 6개 칼럼까지만 출력
            break

    except Exception as e:
        pass

################
#변수간의 상관계수

apple_tmp=apple.copy()
apple_tmp['Label'] = apple['Label'].replace({'1':1, '0':0})
corrmat = apple_tmp.corr()
top_corr_features = corrmat.index[abs(corrmat["Label"])>=0.2]

plt.figure(figsize=(20,20))
g=sns.heatmap(apple_tmp[top_corr_features].corr(),annot=True,fmt='.1f',linewidths=.5,cmap='Blues')


#################
#변수 중요도
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=42, max_depth=6)
clf.fit(x_train, y_train)
feature_importance = clf.feature_importances_


df_fi = pd.DataFrame({'columns':x_train.columns, 'importances':feature_importance})
df_fi = df_fi[df_fi['importances'] > 0] # importance가 0이상인 것만 
df_fi = df_fi.sort_values(by=['importances'], ascending=False)

fig = plt.figure(figsize=(15,7))
ax = sns.barplot(df_fi['columns'], df_fi['importances'])
ax.set_xticklabels(df_fi['columns'], rotation=80, fontsize=13)
plt.tight_layout()
plt.show()

###########################

# 샘플 데이터 표현

plt.scatter(x.iloc[:,0], x.iloc[:,1], c=y, s=30, cmap=plt.cm.Paired)
# 초평면(Hyper-Plane) 표현
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--', '-', '--'])
# 지지벡터(Support Vector) 표현
ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=60, facecolors='r')
plt.show()


##############

#모델링 svm
svm_model =SVC(kernel='rbf')

svm_model.fit(x_train,y_train)

y_pred =svm_model.predict(x_test)
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

#randomforest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

y_pred =rf.predict(x_test)
y_true=y_test

print(classification_report(y_true,y_pred))
print("accuracy:"+str(accuracy_score(y_true, y_pred)))


#knn
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(x_train,y_train)

x_train.shape
y_pred= knn.predict(x_test)

print(classification_report(y_true,y_pred))
print("accuracy:"+str(accuracy_score(y_true, y_pred)))


#macd 지표 제거후 
#train data 생성
x_train=train_data.drop(['Date','Label','Open','High','Low','Close','Volume','Adjusted'],axis=1)
y_train=train_data['Label']
x_train.columns
#test data 생성
test =apple['Date'].str.contains('2019')
test_data =apple[test]

x_test=test_data.drop(['Date','Label','Open','High','Low','Close','Volume','Adjusted'],axis=1)
y_test=test_data['Label']




