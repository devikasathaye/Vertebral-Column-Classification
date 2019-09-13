#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

import pandas as pd
import seaborn as sns
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read data from .dat file to dataframe


get_ipython().system(' git clone https://github.com/devikasathaye/HW1')
df = pd.read_csv('HW1/column_2C.dat', sep=' ', header=None, skiprows=0)
df.columns = ['pelvic incidence', 'pelvic tilt', 'lumbar lordosis angle', 'sacral slope', 'pelvic radius', 'grade of spondylolisthesis', 'class']
df['class'] = df['class'].map({'AB': 1, 'NO': 0}) #Abnormal=1, Normal=0
df.head(10) #Displaying a part of the data


# # (b) Pre-Processing and Exploratory Data Analysis

# ## i. Scatterplots of the independent variables


sns.set(font_scale=2)
g = sns.PairGrid(df, height=5, hue="class",hue_kws={"marker":["o","+"]}, palette=["#FF0000","#0C41CE"], vars=['pelvic incidence','pelvic tilt','lumbar lordosis angle','sacral slope','pelvic radius','grade of spondylolisthesis'])
g = g.map(plt.scatter)
g = g.add_legend()


# ## ii. Boxplots for each of the independent variables


plt.figure(figsize=(20,20))
plt.subplots_adjust(wspace=1)
for i in range(len(df.columns)-1):
    plt.subplot(2,3,i+1)
    sns.boxplot(x='class',y=df.columns[i],data=df)


# ## iii. Splitting the dataset into train data and test data


train1=df[df['class']==0].head(70)
train2=df[df['class']==1].head(140)
train=pd.concat([train1,train2],axis=0) #Training data consists of first 70 rows of class 0 and first 140 rows of class 1
test1=df[df['class']==0][70:]
test2=df[df['class']==1][140:]
test=pd.concat([test1,test2],axis=0)

X_train=train.drop(columns=['class'])
y_train=train['class']
X_test=test.drop(columns=['class'])
y_test=test['class']


# ## (c) Classification using KNN on Vertebral Column Data Set

# ## i. k-nearest neighbors with Euclidean metric


knn = KNeighborsClassifier() #p=2 for Euclidean metric, which is default
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc=(knn.fit(X_train, y_train).score(X_test, y_test))*100

print("Accuracy of the model is {}%".format(acc))


# ## ii. Finding best k, confusion matrix, true positive rate, true negative rate, precision, F-score when k=k*


ErrorTrain=[]
ErrorTest=[]

for i in range(208, 0, -1):
    knn = KNeighborsClassifier(n_neighbors=i)
    ErrorTrain.append(1-(knn.fit(X_train, y_train).score(X_train, y_train)))
    ErrorTest.append(1-(knn.score(X_test, y_test)))


# ## Train error and test error for k=1 to k=208


plt.plot(range(208, 0, -1), ErrorTest, label="TestError")
plt.plot(range(208, 0, -1), ErrorTrain, label="TrainError")
plt.xlim(208,0)
plt.gca().legend(('TestError','TrainError'))
plt.show()


# ## Best k value

print("The minimum test error rate is",round(min(ErrorTest),3),"for k=",208-(ErrorTest.index(min(ErrorTest))))
print("Therefore, k*=",208-(ErrorTest.index(min(ErrorTest))))


# ## KNN for k=k*=4

knn = KNeighborsClassifier(n_neighbors=4) #p=2 for Euclidean metric, which is default
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy=(knn.fit(X_train, y_train).score(X_test, y_test))*100
print("Accuracy of the model is {}%".format(accuracy))


# ## Confusion Matrix

cm = confusion_matrix(y_test,y_pred)
df_cm = pd.DataFrame(cm, index = ["Normal","Abnormal"],
                  columns = ["Normal","Abnormal"])
plt.figure(figsize = (9,7))
x = sns.heatmap(df_cm, annot=True)


# ## True Positive Rate, True Negative Rate, Precision, F-Score

FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]
TN = cm[0,0]

# True Positive Rate
TPR = TP/(TP+FN)
# True Negative Rate
TNR = TN/(TN+FP)
# Precision
Precision = TP/(TP+FP)
# F-Score
FSC = 2*(Precision*TPR)/(Precision+TPR)

print("True Positive Rate:\t",TPR)
print("True Negative Rate:\t",TNR)
print("Precision:\t\t",Precision)
print("F-Score:\t\t",FSC)


# ## iii. Learning Curve

ErrN=[]
ErrkN=[]

for i in range(1, 210, 1):
    n0=math.floor(i/3)
    n1=i-n0
    train_new1=train[train['class']==0].head(n0)
    train_new2=train[train['class']==1].head(n1)
    train_new=pd.concat([train_new1,train_new2], axis=0)

    X_train_new=train_new.drop(columns=['class'])
    y_train_new=train_new['class']

    ErrTest=[]

    #To find Best Test Error Rate to plot learning curve
    for j in range(1, i+1, 5):
        knn = KNeighborsClassifier(n_neighbors=j)
        ErrTest.append(1-(knn.fit(X_train_new, y_train_new).score(X_test, y_test)))
    ErrN.append(min(ErrTest))

    #To find value of k corresponding to the best test error rate for each N
    for k in range(1, i+1, 5):
        knn = KNeighborsClassifier(n_neighbors=k, p=2)
        ErrTest.append(1-(knn.fit(X_train_new, y_train_new).score(X_test, y_test)))
    ErrkN.append(ErrTest.index(min(ErrTest)))


#Learning curve
plt.plot(range(1, 210, 1), ErrN)
plt.xlabel("Value of N")
plt.ylabel("Best Test Error Rate")
plt.show()
#--------------------------------------
plt.plot(range(1, 210, 1), ErrkN)
plt.xlabel("Value of N")
plt.ylabel("Value of k")
plt.show()


# ## (d) Using different distance metrics

# ## i. Minkowski Distance

# ## A. Manhattan Distance

BestTestError=[]
BestK=[]
ErrorTestMan=[]

step=5
for i in range(1, 197, step):
    knn = KNeighborsClassifier(n_neighbors=i, p=1)
    ErrorTestMan.append(1-(knn.fit(X_train, y_train).score(X_test, y_test)))
BestTestError.append(min(ErrorTestMan))
BestK.append(1+step*ErrorTestMan.index(min(ErrorTestMan)))
k=1+step*ErrorTestMan.index(min(ErrorTestMan))
print("k=",k)
print("Best Test Error with Manhattan Distance=",min(ErrorTestMan))


# ## B. k*=6, log10(p)={0.1, 0.2, 0.3, ..., 1}. Finding best log10(p)

ErrorTestP=[]

i=0.1
while(i<1.1):
    knn = KNeighborsClassifier(n_neighbors=6, p=math.pow(10,i))
    ErrorTestP.append(1-(knn.fit(X_train, y_train).score(X_test, y_test)))
    i=round(i+0.1,1)
BestTestError.append(min(ErrorTestP))
BestK.append(6)
i=0.1+(ErrorTestP.index(min(ErrorTestP)))/10
print("Best log10(p)=",i)
print("Best Test Error=",min(ErrorTestP))
pval=math.pow(10,i)
print("Value of p=",pval)


# ## C. Chebyshev Distance(p->∞)

ErrorTestCheb=[]
step=5

for i in range(1, 197, step):
    knn = KNeighborsClassifier(n_neighbors=i, metric='chebyshev')
    ErrorTestCheb.append(1-(knn.fit(X_train, y_train).score(X_test, y_test)))
BestTestError.append(min(ErrorTestCheb))
BestK.append(1+step*ErrorTestCheb.index(min(ErrorTestCheb)))
k=1+step*ErrorTestCheb.index(min(ErrorTestCheb))
print("k=",k)
print("Best Test Error with Chebyshev Distance=",min(ErrorTestCheb))


# ## ii. Mahalanobis Distance

X=pd.concat([X_train, X_test])

ErrorTestMaha=[]
step=5

for i in range(1, 197, step):
    knn = KNeighborsClassifier(n_neighbors=i, metric='mahalanobis', metric_params={'V':X.cov()})
    ErrorTestMaha.append(1-(knn.fit(X_train, y_train).score(X_test, y_test)))
BestTestError.append(min(ErrorTestMaha))
BestK.append(1+step*ErrorTestMaha.index(min(ErrorTestMaha)))
k=1+step*ErrorTestMaha.index(min(ErrorTestMaha))
print("k=",k)
print("Best Test Error with Mahalanobis Distance=",min(ErrorTestMaha))


# ## Comparison Table

Errors=[]
DistMetric=[]

DistMetric.append('Manhattan')
DistMetric.append('log10(p)=0.6')
DistMetric.append('Chebyshev')
DistMetric.append('Mahalanobis')

Errors.append(DistMetric)
Errors.append(BestK)
Errors.append(BestTestError)

Errors=list(map(list,zip(*Errors)))

ErrorDF=pd.DataFrame(Errors)
ErrorDF.columns=['Distance Metric', 'k*', 'Best Test Error Rate']

ErrorDF


# ## (e) Weighted Decision

ErrorTestM=[]
ErrorTestE=[]
ErrorTestC=[]
Err=[]
BestTestErrorMEC=[]
Best_K=[]
DistanceMetric=[]
step=5

for i in range(1, 197, step):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', p=1) #Manhattan Distance
    ErrorTestM.append(1-(knn.fit(X_train, y_train).score(X_test, y_test)))
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', p=2) #Euclidean Distance
    ErrorTestE.append(1-(knn.fit(X_train, y_train).score(X_test, y_test)))
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', metric='chebyshev') #Chebyshev Distance
    ErrorTestC.append(1-(knn.fit(X_train, y_train).score(X_test, y_test)))

BestTestErrorMEC.append(min(ErrorTestM))
BestTestErrorMEC.append(min(ErrorTestE))
BestTestErrorMEC.append(min(ErrorTestC))

Best_K.append(1+step*ErrorTestM.index(min(ErrorTestM)))
Best_K.append(1+step*ErrorTestE.index(min(ErrorTestE)))
Best_K.append(1+step*ErrorTestC.index(min(ErrorTestC)))

DistanceMetric.append('Euclidean')
DistanceMetric.append('Manhattan')
DistanceMetric.append('Chebyshev')


Err.append(DistanceMetric)
Err.append(Best_K)
Err.append(BestTestError)

Err=list(map(list,zip(*Err)))

ErrDF=pd.DataFrame(Err)
ErrDF.columns=['Distance Metric', 'k*', 'Best Test Error Rate']

ErrDF


# ## (f) Lowest Training Error Rate

ErrorTrain_Euc=[]
ErrorTest_Euc=[]
step=5

for i in range(1, 197, step):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', p=2) #Euclidean Distance
    ErrorTrain_Euc.append(1-(knn.fit(X_train, y_train).score(X_train, y_train)))
    ErrorTest_Euc.append(1-(knn.fit(X_train, y_train).score(X_test, y_test)))

print("Lowest training error rate is",min(ErrorTrain_Euc))


# The minimum training error rate achieved above is 0.0. It cannot be lower than 0.
# Hence, the lowest training error rate achieved is 0.
