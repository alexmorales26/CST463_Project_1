# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:51:36 2018

@author: Alexander Morales, Nolan Guzman, & Alyssia Goodwin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dat = pd.read_csv("https://raw.githubusercontent.com/alexmorales26/CST463_Project_1/master/default_cc_train.csv")

dat.isnull().sum()
dat.rename(columns={'default.payment.next.month':'default'},inplace=True)
dat.drop('ID',axis=1,inplace=True)

dat['AvgStatement'] = dat.iloc[:,11:17].mean(axis=1)

#For eduation the values of 0, 5, and 6 are not explained on the attribute information
#   within the documentation, thus it was put into the other cateogry 
dat['EDUCATION'].value_counts()

dat['EDUCATION'].where(dat['EDUCATION']<4,4,inplace=True)
dat['EDUCATION'].where(dat['EDUCATION']!=0,4,inplace=True)


dat['EDUCATION'].value_counts()

dat.describe()
dat.dtypes

#Male = 1, Female = 2.
#More Females than male 
dat['SEX'].value_counts()


plt.hist(dat['EDUCATION'])


#Looking at the plots it is able to be seen the higher education a person has
#such as the distribution for graduate school has a higher balance compared to 
#the other education. 

sns.violinplot(dat['EDUCATION'], dat['LIMIT_BAL'])

sns.countplot(dat['EDUCATION'])


plt.hist(dat['AGE'])

temp = dat[dat['AvgStatement']>dat['LIMIT_BAL']]
plt.hist(temp['SEX'])

sns.barplot(y = 'AGE',x='default',hue='EDUCATION',data=dat)

