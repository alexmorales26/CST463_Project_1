# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:51:36 2018

@author: Alexander Morales, Nolan Guzman, & Alyssia Goodwin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
dat = pd.read_csv("https://raw.githubusercontent.com/alexmorales26/CST463_Project_1/master/default_cc_train.csv")

#sum of the null values
dat.isnull().sum()
#describes the data
dat.describe()
#gives all the types
dat.dtypes
#Renames deafult payment next month to default 
dat.rename(columns={'default.payment.next.month':'default'},inplace=True)
#drop the ID 
dat.drop('ID',axis=1,inplace=True)

#Average statement per person into a column 
dat['AvgStatement'] = dat.iloc[:,11:17].mean(axis=1)

dat['OnTimePayment'] = np.ceil(dat.iloc[:,5:11].mean(axis=1))
dat['OnTimePayment'].where(dat['OnTimePayment']!=-0,0,inplace=True)

dat['AvgPayAmt'] = dat.iloc[:,17:23].mean(axis=1)

dat['PayDiff'] = dat['AvgStatement']-dat['AvgPayAmt']
#For eduation the values of 0, 5, and 6 are not explained on the attribute information
#   within the documentation, thus it was put into the other cateogry 
dat['EDUCATION'].value_counts()

dat[dat['EDUCATION']==0] = 4 
dat[dat['EDUCATION']==5] = 4
dat[dat['EDUCATION']==6] = 4

dat['EDUCATION'].value_counts()


#Male = 1, Female = 2.
#More Females than male 
dat['SEX'].value_counts()

#Most people have university education or higher
plt.hist(dat['EDUCATION'])

#Looking at the plots it is able to be seen the higher education a person has
#such as the distribution for graduate school has a higher balance compared to 
#the other education. 

sns.violinplot(dat['EDUCATION'], dat['LIMIT_BAL'])


plt.hist(dat['AGE'])


temp = dat[dat['AvgStatement']>dat['LIMIT_BAL']]
plt.hist(temp['SEX'])

sns.barplot(y = 'AGE',x='default',hue='EDUCATION',data=dat)

# svm crap starts here =====================================================
# X includes derived features
X= dat[["AvgStatement","EDUCATION","LIMIT_BAL","PayDiff"]]
# what we want to predict
y= dat["default"]
# we normalize X for better performance
X = normalize(X)
# split the data with 70% being training data, and 30% for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# default SVC function call
svm_clf = SVC()
# fit the SVC with the training params X and y training datasets
svm_clf.fit(X_train,y_train)
# once its trained give it X test data to predict what y could be
svm_clf.predict(X_test)
# compare the accuracy scores and print result
print(svm_clf.score(X_test,y_test)) 


# grid search => best hyperparams will be outputted and best prediction features as well

param_grid  = [{'degree':[3,5,7],'kernel':['poly','rbf','sigmoid']}]
GridSearch = GridSearchCV(svm_clf,param_grid,cv=5,scoring='accuracy',n_jobs=-1)
GridSearch.fit(X_train,y_train)
print(GridSearch.best_params_)
print(GridSearch.best_estimator_)


# new svc and then re-apply SVC will grid search hyperparams
svm_clf_new = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='poly',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

svm_clf_new.fit(X_train,y_train)
svm_clf_new.predict(X_test)
# should be the same SVC score as above, meaning that is as good as SVC will get
print(svm_clf_new.score(X_test,y_test))


