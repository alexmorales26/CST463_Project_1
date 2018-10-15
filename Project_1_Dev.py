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



dat.describe()
dat.dtypes
#Male = 1, Female = 2.
#More Females than male 
dat['SEX'].value_counts()


#For eduation the values of 0, 5, and 6 are not explained on the attribute information
#   within the documentation, thus it was put into the other cateogry 
dat['EDUCATION'].value_counts()

dat[dat['EDUCATION']==0] = 4
dat[dat['EDUCATION']==5] = 4
dat[dat['EDUCATION']==6] = 4

dat['EDUCATION'].value_counts()

#dat['EDUCATION'].plot(kind='bar')

plt.hist(dat['EDUCATION'])


#Looking at the plots it is able to be seen the higher education a person has
#such as the distribution for graduate school has a higher balance compared to 
#the other education. 

sns.violinplot(dat['EDUCATION'], dat['LIMIT_BAL'])

sns.countplot(dat['EDUCATION'])