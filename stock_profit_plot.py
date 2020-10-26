# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:57:44 2020

@author: user
"""

import pandas as pd
import numpy as np
stock =pd.read_csv('C:/Users/user/Desktop/study/stock_2019_position.csv')


#마지막 강제 청산
stock['position'][251] ='sell'

stock=stock.drop(stock.columns[0],axis='columns')

#date 날짜형으로 변경
from datetime import datetime
import time

time_format="%Y-%m-%d"

stock['Date']=pd.to_datetime(stock['Date'],format=time_format)

#buy => close
#sell => close


#buy label index

buy_position = stock['position']=='buy'

buy_label= stock[buy_position]


#sell label index
sell_position =stock['position'] =='sell'

sell_label =stock[sell_position]

#hold label index
hold_position= stock['position']=='holding'

hold_label=stock[hold_position]

#no action label index
no_position=stock['position']=='no action'

no_label=stock[no_position]

#profit 생성
stock['profit']=None


for i in range(len(stock['position'])):
    if stock['position'][i] == 'holding':
        stock['profit'][i] =0
    elif stock['position'][i] =='no action':
        stock['profit'][i] =0
    elif stock['position'][i]=='buy':
        stock['profit'][i]=0
    else:
        print(i)
        
####
buy_position = stock['position']=='buy'

buy_label= stock[buy_position]


#sell label index
sell_position =stock['position'] =='sell'

sell_label =stock[sell_position]

a =buy_label['Close'].reset_index(drop=True)
b =sell_label['Close'].reset_index(drop=True)

profit=b-a

profit =list(profit)


stock.loc[stock['position']=='sell','profit'] = profit

####
stock['profit_cumsum']=stock['profit'].cumsum()

###stock_profit plot
import matplotlib.pyplot as plt
import seaborn as sns
plt.title('stock_profit')
plt.plot(stock[['Date']],stock[['profit_cumsum']],color='blue')
plt.show()


