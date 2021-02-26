#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:24:32 2020

@author: omertas
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
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

#all features but exclude class labels
X = brain[clmns[0:-1]].values

le = LabelEncoder ()
Y = le.fit_transform(brain["death01"].values)

#split data 68% to 32%
X_train, X_test, Y_train, Y_test = \
    train_test_split(X,Y,train_size = 0.68, stratify = Y, random_state=9)



#decision tree    
tree_classifier = tree.DecisionTreeClassifier(criterion='entropy')
tree_classifier = tree_classifier.fit(X, Y)
prediction = tree_classifier.predict(X_test)
error_rate = np.mean(prediction!=Y_test)
cfm_tree = confusion_matrix(Y_test, prediction)

#print function 
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
    
prnt_all(cfm_tree)    


print("\n")



print("Random instances are classified")

#lets choose some random instances and make predictions
items = [range(1,5),range(1,6),range(1,5),range(1,3),range(1,5),
         range(1,4),range(1,4),range(1,4),range(1,3),range(1,4),
         range(1,7),range(1,3),range(20,76),range(2,4), range(1,3)]


for j in range(95,100):
    random.seed(j)
    new_item = []
    for i in items:
        x = random.sample(i,1)
        new_item += x
    new_item = np.asmatrix(new_item)
    print(tree_classifier.predict(new_item))