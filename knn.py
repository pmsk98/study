# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 14:19:54 2020

@author: pmsk9
"""
#특정 농구선수의 포지션을 예측
import pandas as pd

df=pd.read_csv('https://raw.githubusercontent.com/wikibook/machine-learning/2.0/data/csv/basketball_stat.csv')

df.head()

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#스틸,2점슛 데이터 시각화
sns.lmplot('STL','2P',data=df,fit_reg=False,
          scatter_kws={'s':150},
          markers=['o','x'],
          hue='Pos')

plt.title('STL and 2P in 2d plane')
