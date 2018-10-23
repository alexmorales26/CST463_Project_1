# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:51:36 2018

@author: Alexander Morales, Nolan Guzman, & Alyssia Goodwin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
import itertools
from sklearn.ensemble import RandomForestRegressor

import warnings

warnings.filterwarnings('ignore')


dat = pd.read_csv("https://raw.githubusercontent.com/alexmorales26/CST463_Project_1/master/default_cc_train.csv")

#sum of the null values
dat.isnull().sum()
#describes the data
dat.describe()
#gives all the types
dat.dtypes
#Renames deafult payment next month to default 
dat.rename(columns={'default.payment.next.month':'default'},inplace=True)
dat.drop('ID',axis=1,inplace=True)

#Average statement per person into a column 
dat['AvgStatement'] = dat.iloc[:,11:17].mean(axis=1)
#Ontime round, taking the average of each row on their payments they have made if they are ontime,
#not paid, paid in full, or dealyed for a number of months
dat['OnTimePayment'] = np.ceil(dat.iloc[:,5:11].mean(axis=1))
dat['OnTimePayment'].where(dat['OnTimePayment']!=-0,0,inplace=True)
#Creates a column with the average amount as an indiviudal they payed
dat['AvgPayAmt'] = dat.iloc[:,17:23].mean(axis=1)
#Creates a column with the differnce between the statment and the amount the indivdual pays.
dat['PayDiff'] = dat['AvgStatement']-dat['AvgPayAmt']

dat['OnTimePayment'] = np.ceil(dat.iloc[:,5:11].mean(axis=1))
dat['OnTimePayment'].where(dat['OnTimePayment']!=-0,0,inplace=True)

dat['AvgPayAmt'] = dat.iloc[:,17:23].mean(axis=1)

dat['PayDiff'] = dat['AvgStatement']-dat['AvgPayAmt']
#For eduation the values of 0, 5, and 6 are not explained on the attribute information
#   within the documentation, thus it was put into the other cateogry 
dat['EDUCATION'].value_counts()


dat['EDUCATION'].where(dat['EDUCATION']<4,4,inplace=True)
dat['EDUCATION'].where(dat['EDUCATION']!=0,4,inplace=True)



sns.violinplot(dat['EDUCATION'], dat['LIMIT_BAL'])
plt.title("Education of Individual vs Limit Balance")
plt.xlabel("Education")
plt.ylabel("Limit Balance")

sns.countplot(x='default',hue='EDUCATION',data=dat)
plt.title("Count of Those who Defaulted on Credit, Sorted by Education")

dat['SEX'].value_counts()

sns.jointplot(dat['PayDiff'],dat['AvgPayAmt'],kind="reg")

sns.jointplot(dat['OnTimePayment'], dat['AvgPayAmt'])
sns.countplot(dat['EDUCATION'])

plt.hist(dat['AGE'])

#After finding the average amount of people statements over 6 months, we wanted
#to know which sex was more likely surpass their credit limit, and of those, who 
#paid them off on time
temp = dat[dat['AvgStatement']>dat['LIMIT_BAL']]
sns.countplot(temp['SEX'],hue=temp['OnTimePayment'])
plt.title('Sex of High Credit Users, Sorted by On Time Payment')

#Looking at the plots it is able to be seen the higher education a person has
#such as the distribution for graduate school has a higher balance compared to 
#the other education. 

sns.violinplot(dat['EDUCATION'], dat['LIMIT_BAL'])


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

temp = dat[dat['AvgStatement']>dat['LIMIT_BAL']]
sns.countplot(temp['SEX'],hue=temp['OnTimePayment'])
plt.title('Sex of High Credit Users, Sorted by On Time Payment')



payment = sns.jointplot(x='PayDiff',y='OnTimePayment',data=dat,kind='hex',stat_func=None,ylim=(-2,3),xlim=(-50000,300000))
plt.title('Payment Difference by On Time Payments',loc='left')
legend = plt.colorbar()
legend.set_label('Count')

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
svm_clf.score(X_test,y_test)





#####RANDOM FOREST#####
X= dat.drop('default',axis=1)
y = dat["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
rf = RandomForestRegressor(n_estimators = 10000, random_state = 42)
rf.fit(X_train, y_train)
rf.predict(X_test)
rf.score(X_test, y_test)

importances = list(rf.feature_importances_)
importances



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
temppred=svm_clf_new.predict(X_test)
# should be the same SVC score as above, meaning that is as good as SVC will get
print(svm_clf_new.score(X_test,y_test))
###
pd.crosstab(y_test, temppred, rownames=['True'], colnames=['Predicted'], margins=True)


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    

cmatrix = confusion_matrix(y_test,temppred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cmatrix, ['0','1'],
                      title='Confusion matrix')

# adaBoost Classifier
ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=4),
        n_estimators=200,
        algorithm="SAMME.R", 
        learning_rate=0.5
        )
ada_clf.fit(X_train,y_train)
print(ada_clf.feature_importances_)



### all 25 features ada boost
A= dat.drop('default',axis=1)
y= dat["default"]
# we normalize X for better performance
A = normalize(A)
# split the data with 70% being training data, and 30% for test
A_train, A_test, y_train, y_test = train_test_split(A, y, test_size=0.3)
ada_clf2 = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=4),
        n_estimators=200,
        algorithm="SAMME.R", 
        learning_rate=0.5
        )
ada_clf2.fit(A_train,y_train)
print(ada_clf2.feature_importances_)







