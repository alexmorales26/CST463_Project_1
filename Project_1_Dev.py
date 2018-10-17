# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:51:36 2018

@author: Alexander Morales, Nolan Guzman, & Alyssia Goodwin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dat = pd.read_csv("https://raw.githubusercontent.com/alexmorales26/CST463_Project_1/master/default_cc_train.csv")

dat.isnull().sum()
dat.rename(columns={'default.payment.next.month':'default'},inplace=True)
dat.drop('ID',axis=1,inplace=True)

dat['AvgStatement'] = dat.iloc[:,11:17].mean(axis=1)
dat['OnTimePayment'] = np.ceil(dat.iloc[:,5:11].mean(axis=1))
dat['OnTimePayment'].where(dat['OnTimePayment']!=-0,0,inplace=True)
dat['AvgPayAmt'] = dat.iloc[:,17:23].mean(axis=1)
dat['PayDiff'] = dat['AvgStatement']-dat['AvgPayAmt']

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

#After finding the average amount of people statements over 6 months, we wanted
#to know which sex was more likely surpass their credit limit, and of those, who 
#paid them off on time
temp = dat[dat['AvgStatement']>dat['LIMIT_BAL']]
sns.countplot(temp['SEX'],hue=temp['OnTimePayment'])
plt.title('Sex of High Credit Users, Sorted by On Time Payment')

#We have simplified our education column and we would like to see if there
#education is a good justifier of someone defaulting on their credit.
sns.countplot(x='default',hue='EDUCATION',data=dat)
plt.title("Count of Those who Defaulted on Credit, Sorted by Education")

#With our new PayDiff column, we are trying to see if there is any sort of corellation
#between the amount of leftover debt and their average duly payments.
payment = sns.jointplot(x='PayDiff',y='OnTimePayment',data=dat,kind='hex',stat_func=None,ylim=(-2,3),xlim=(-50000,300000))
plt.title('Payment Difference by On Time Payments',loc='left')
legend = plt.colorbar()
legend.set_label('Count')

# this swarm plot uses X as a quote on quote categorical feature. Therefore, we want to
# know the avgStatement of each age contained in the datset. This will give a better picture
# to see what age group has this biggest avgStatement
sns.swarmplot(x="AGE",y="AvgStatement", data = dat)
#train split and SVM crapp
# =============================================================================
# X = dat
# y= dat['default']
# 
# train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.3)
# 
# svm_clf = SVC()
# svm_clf.fit(train_X,train_y)
# X_1 = svm_clf.predict(test_X)
# =============================================================================


