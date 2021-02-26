#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:23:45 2020

@author: omertas
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import random 


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
    
log_reg_classifier = LogisticRegression() 
log_reg_classifier.fit(X_train ,Y_train)
prediction = log_reg_classifier.predict(X_test)
accuracy = np.mean(prediction == Y_test)
cfm = confusion_matrix(Y_test, prediction)
TPR = cfm[0][0]/sum(cfm[0])
print("Accuracy is", str(round(accuracy*100,2)) + "%")
print("TPR is", str(round(TPR*100,2)) + "%")

cfm_knn = confusion_matrix(Y_test, prediction)

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
    
prnt_all(cfm_knn)    

print("\n")



print("Random instances are classified")


#Now we will predict random inputs

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
    print(log_reg_classifier.predict(new_item))