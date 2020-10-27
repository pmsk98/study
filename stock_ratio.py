# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:07:03 2020

@author: user
"""

import pandas as pd
import numpy as np
stock =pd.read_csv('C:/Users/user/Desktop/study/stock_2019_position.csv')


stock=stock.drop(stock.columns[0],axis='columns')


#profit 생성
stock['profit']=None

#다음날 수정 종가를 가져오게 생성
for i in range(len(stock)):
    if stock['position'][i]=='buy':
        stock['profit'][i]=stock['Adjusted'][i+1]
    elif stock['position'][i]=='sell':
        stock['profit'][i]=stock['Adjusted'][i+1]
    else:
        print(i)
        
#마지막 강제 청산        
stock['position'][251] ='sell'

stock['profit'][251]=stock['Adjusted'][251]


####
buy_position = stock['position']=='buy'

buy_label= stock[buy_position]


#sell label index
sell_position =stock['position'] =='sell'

sell_label =stock[sell_position]


        

profit=b-a

######
#거래 횟수

ratio =pd.DataFrame(data={'stock ticker':['AAPL'],'NO.trades':[None],'Win%':[None],'Average gain($)':[None],'Average loss($)':[None],'Payoff ratio':[None],'Profit factor':[None]})

ratio['NO.trades']= len(profit)

#승률

win=profit.groupby('average').size()[0]/len(profit)
ratio['Win%']=win



#평균 수익

profit =pd.DataFrame(profit)

profit['average']= None


for i in range(len(profit)):
    if profit['Adjusted'][i] > 0:
        profit['average'][i]='gain'
    else:
        profit['average'][i]='loss'



for i in range(len(profit)):
    if profit['Adjusted'][i] < 0:
        profit['Adjusted'][i]=profit['Adjusted'][i] *-1
    else:
        print(i)
        


gain_loss=profit.groupby('average').mean()

gain_loss=pd.DataFrame(gain_loss)

ratio['Average gain($)']=gain_loss['Adjusted'][0]
#평균 손실


ratio['Average loss($)']=gain_loss['Adjusted'][1]

#payoff ratio
ratio['Payoff ratio']=gain_loss['Adjusted'][0]/gain_loss['Adjusted'][1]

#profit factor

gain_loss=profit.groupby('average').sum()

gain_loss=pd.DataFrame(gain_loss)


ratio['Profit factor']=gain_loss['Adjusted'][0]/gain_loss['Adjusted'][1]


ratio.to_csv('AAPL_ratio.csv')


