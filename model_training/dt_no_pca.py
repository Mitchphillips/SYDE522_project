# -*- coding: utf-8 -*-
"""SYDE522_DT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mBe_9Z1YOjCMRaJWzOmGyrA6pZaluExP
"""

# import standard libraries
from __future__ import absolute_import, division, print_function, unicode_literals

from shutil import copyfile

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn import tree

from sklearn.metrics import classification_report
import itertools
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import preprocessing

# load train data
train_data = pd.read_csv('/content/drive/My Drive/SYDE 522 Project/MoreManipulatedData.csv',usecols=lambda x: 'PLAYER_URL' not in x)

# different classes
train_data.BUCKET.unique()

# convert string values to numerical data
def convert(data):
    number = preprocessing.LabelEncoder()
    data['POS'] = number.fit_transform(data.POS)
    data['LEAGUE'] = number.fit_transform(data.LEAGUE)
    data['FIRST_JUNIOR_YEAR'] = number.fit_transform(data.FIRST_JUNIOR_YEAR)
    data['DOB'] = number.fit_transform(data.DOB)
    data['NATIONALITY'] = number.fit_transform(data.NATIONALITY)
    data['SHOOTS'] = number.fit_transform(data.SHOOTS)
    data=data.fillna(-999)
    return data

# convert string values in data to numerical classes
train_data=convert(train_data)

# Separating the data and the labels
X = np.asarray(train_data[train_data.columns[:-1]])
y = np.asarray(train_data.BUCKET)

# Splitting the data into the train and the test sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
sss.get_n_splits(X, y)

train_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

print('Training data: \n',X)
print('\n')
print('Training labels: \n',y_train)

# Decision Tree Classifier (has to be able to deal with floats)
DT = tree.DecisionTreeClassifier()

# fit the classifier using the training data
DT = DT.fit(X_train, y_train)

# Predict the test class labels using the trained DT classifier 
y_pred = DT.predict(X_test)

# print accuracy of the classifier
print(classification_report(y_test, y_pred))