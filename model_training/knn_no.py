# import standard libraries
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import itertools

from sklearn.datasets import load_iris
from sklearn import tree, preprocessing
from sklearn.metrics import classification_report, confusion_matrix,f1_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from shutil import copyfile

# load train data
train_data = pd.read_csv('../data/MoreManipulatedData_bucket_last_col.csv',usecols=lambda x: 'PLAYER_URL' not in x)

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
train_data = convert(train_data)

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

ks = []
f1s_micro = []
f1s_macro = []
f1s_weighted = []

K = 2
while K < 100:
	print(K)
	# K Nearest Neighbors Classifier (has to be able to deal with floats)
	KNN = KNeighborsClassifier(K)

	# fit the classifier using the training data
	KNN = KNN.fit(X_train, y_train)

	# Predict the test class labels using the trained KNN classifier 
	y_pred = KNN.predict(X_test)


	ks.append(K)
	f1s_macro.append(f1_score(y_test, y_pred,average='macro'))
	f1s_micro.append(f1_score(y_test, y_pred,average='micro'))
	f1s_weighted.append(f1_score(y_test, y_pred,average='weighted'))

	# print accuracy of the classifier
	# print(classification_report(y_test, y_pred))

	K = K + 1

fig, axes = plt.subplots(1, 1)

axes.plot(ks,f1s_micro)
axes.plot(ks,f1s_macro)
axes.plot(ks,f1s_weighted)
labels = ["Micro average","Macro average","Weighted average"]
axes.legend(axes.get_lines(), labels, loc=7)
plt.ylabel('Accuracy')
plt.xlabel('Value of K')
plt.title('Effect of K on accuracy')
plt.show()





