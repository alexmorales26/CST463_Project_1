# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:51:36 2018

@author: Alexander Morales, Nolan Guzman, & Alyssia Goodwin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dat = pd.read_csv("default_cc_train.csv")
dat.rename(columns={'default.payment.next.month':'default'},inplace=True)
dat.drop('ID',axis=1,inplace=True)

dat['AvgStatement'] = dat.iloc[:,11:17].mean(axis=1)

dat.isnull().sum()
plt.hist(dat['AGE'])

temp = dat[dat['AvgStatement']>dat['LIMIT_BAL']]
plt.hist(temp['SEX'])

sns.barplot(y = 'AGE',x='default',hue='EDUCATION',data=dat)