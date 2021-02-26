


#we will ony perform multiple supervised learning
#classifier
"""
###############################################################
#Data preprocessing
brain = pd.read_csv("data.csv")
clmns = brain.columns


#remove row with missing class values 
class_index_nan = \
    brain['death01'].index[brain['death01'].apply(np.isnan)]

brain = brain.drop(brain.index[class_index_nan])

#handling missing values in attributes
#catagorical variable missing values will be filled with mode
#continuous varible missing values will be replaced with mean
#We have only one continuous attribute and which doesn't have 
#missing values so we can apply follwoing function to fill missing
#values
#Since we are missing values only from catagorical varaible
#we will replace missing values with modes
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(brain)
brain = imputer.transform(brain.values)
brain = pd.DataFrame(brain)

brain.columns = clmns

brain.to_csv("cleandata.csv")

#preprocessing is done
"""

###############################################################


###############################################################

#We will continue with cleaned data cleandata.csv instead of data.csv

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
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
le = LabelEncoder ()
Y = le.fit_transform(brain["death01"].values)


X_train, X_test, Y_train, Y_test = \
    train_test_split(X,Y,train_size = 0.68, stratify = Y, random_state=9)
    
tree_classifier = tree.DecisionTreeClassifier(criterion='entropy')
tree_classifier = tree_classifier.fit(X, Y)
prediction = tree_classifier.predict(X_test)
error_rate = np.mean(prediction!=Y_test)
cfm_tree = confusion_matrix(Y_test, prediction)

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




#Following graphs will give you an oppurtunity
#to visualize how attributes and class label are related
#We will use sctter plot any attributes that has continuous
#varaible and use violin for catogrical vs catagorical

#for scatter plot make sure at least one of your attribute is
#continuous. Scatter plot of catog vs catog my not give you an idea
#print(brain.columns)
#age_at_initial_pathologic is the only continuous attribute by the way

#I will visualize age vs death
#If you are would liek to see how other attributed are realted you
#can simply chage the name of attriibute
fig, ax = plt.subplots(1,figsize =(7,5))
plt.scatter(brain["age_at_initial_pathologic"],brain["death01"])


x_label = "age_at_initial_pathologic"
y_label = "death01"

plt.legend()
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.tight_layout()
plt.show()

#Since sctter plot doesn't explain catog vs catog we will
#use violin
#If you would like to see how other catog attributes realted
#just use a different name
sns.catplot("COCCluster","death01",
            data=brain, kind = "violin")



