#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:29:31 2020

@author: omertas
"""


import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random 


def prnt_all(confusion):
    TP = confusion[0][0]
    FP = confusion[1][0]
    TN = confusion[1][1]
    FN = confusion[0][1]
    print("TP is", TP)
    print("FP is", FP)
    print("TN is", TN)
    print("FN is", FN)
    print("TPR is", str(round(TP/(TP+FN)*100,2))+"%")
    print("TNR is", str(round(TN/(TN+FP)*100,2))+"%")
    print("Accuracy is", str(round((TP+TN)/(TP+FN+TN+FP)*100,2))+"%")
    
  


brain = pd.read_csv("cleandata.csv")
#Data already cleaned but still need to work on little bit more

#this column is just order of rows not part of data
del brain['Unnamed: 0']

#this id of each picture 
#has no meaning to data
del brain["Patient"]
#this column only has value of 1. meaningless to class label
del brain["tumor_tissue_site"]

clmns = brain.columns.to_list()
X = brain[clmns[0:-1]].values
scaler = StandardScaler() 
scaler.fit(X)
X = scaler.transform(X)
le = LabelEncoder ()
Y = le.fit_transform(brain["death01"].values)


X_train, X_test, Y_train, Y_test = \
    train_test_split(X,Y,train_size = 0.68, stratify = Y, random_state=9)


svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train,Y_train)
predicted_linear_svm = svm_classifier.predict(X_test)
error_rate = np.mean(predicted_linear_svm!=Y_test)
accuracy = metrics.accuracy_score(Y_test,predicted_linear_svm)
cfm_svm_linear = confusion_matrix(Y_test,predicted_linear_svm)

prnt_all(cfm_svm_linear)


svm_classifier = svm.SVC(kernel='rbf')
svm_classifier.fit(X_train,Y_train)
predicted = svm_classifier.predict(X_test)
error_rate = np.mean(predicted!=Y_test)
accuracy = metrics.accuracy_score(Y_test,predicted)
cfm_svm_gauss = confusion_matrix(Y_test,predicted)

prnt_all(cfm_svm_gauss)



svm_classifier = svm.SVC(kernel='poly', degree = 5)
svm_classifier.fit(X_train,Y_train)
predicted = svm_classifier.predict(X_test)
error_rate = np.mean(predicted!=Y_test)
accuracy = metrics.accuracy_score(Y_test,predicted)
cfm_svm_cubic = confusion_matrix(Y_test,predicted)

prnt_all(cfm_svm_cubic )


svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train,Y_train)
predicted_linear_svm = svm_classifier.predict(X_test)
error_rate = np.mean(predicted_linear_svm!=Y_test)
accuracy = metrics.accuracy_score(Y_test,predicted_linear_svm)
cfm_svm_linear = confusion_matrix(Y_test,predicted_linear_svm)

print("\n")



print("Random instances are classified")

items = [range(1,5),range(1,6),range(1,5),range(1,3),range(1,5),
         range(1,4),range(1,4),range(1,4),range(1,3),range(1,4),
         range(1,7),range(1,3),range(20,76),range(2,4), range(1,3)]



for j in range(95,100):
    random.seed(j)
    new_item = []
    for i in items:
        x = random.sample(i,1)
        new_item += x
    new_item = scaler.transform(np.asmatrix(new_item))
    print(svm_classifier.predict(new_item))