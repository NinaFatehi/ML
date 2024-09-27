import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 

## Read the input data 
drug = pd.read_csv("drug200.csv")
drug.head()
drug["Drug"].value_counts()
drug["BP"].value_counts()
drug["Cholesterol"].value_counts()

## Convert the categorical attributes into numerical values
from sklearn.preprocessing import LabelEncoder
drug["Sex"] = LabelEncoder().fit_transform(drug["Sex"])
drug["BP"] = LabelEncoder().fit_transform(drug["BP"])
drug["Cholesterol"] = LabelEncoder().fit_transform(drug["Cholesterol"] )
drug["Drug"] = LabelEncoder().fit_transform(drug["Drug"])

## Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(drug.drop(["Drug"], axis=1), drug["Drug"], test_size=0.3, random_state=3)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

## Build, train and evaluate the classifier
from sklearn.tree import DecisionTreeClassifier
drugtree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugtree.fit(X_train, y_train)
yhat = drugtree.predict(X_test)
df = pd.DataFrame({"True Label":y_test, "Predicted Label":yhat})
accuracy = metrics.accuracy_score(y_test, yhat)


## Visualize the tree
import sklearn.tree as tree
tree.plot_tree(drugtree)
plt.show()




