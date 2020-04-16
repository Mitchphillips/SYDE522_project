
# import standard libraries
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import itertools

from sklearn.datasets import load_iris
from sklearn import tree, preprocessing
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_fscore_support,f1_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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

import warnings

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning

# different learning rate schedules and momentum parameters
# params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
#            'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
#            'nesterovs_momentum': False, 'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
#            'nesterovs_momentum': True, 'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
#            'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
#            'nesterovs_momentum': True, 'learning_rate_init': 0.2},
#           {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
#            'nesterovs_momentum': False, 'learning_rate_init': 0.2},
#           {'solver': 'adam', 'learning_rate_init': 0.01}]

# labels = ["constant learning-rate", "constant with momentum",
#           "constant with Nesterov's momentum",
#           "inv-scaling learning-rate", "inv-scaling with momentum",
#           "inv-scaling with Nesterov's momentum", "adam"]

params = [{'solver': 'adam', 'learning_rate_init': 0.01 },
          {'solver': 'adam', 'learning_rate_init': 0.001 },
          {'solver': 'adam', 'learning_rate_init': 0.0001 },
          {'solver': 'adam', 'learning_rate_init': 0.00001 },
          {'solver': 'adam', 'learning_rate_init': 0.000001 }]

labels = ['learning_rate = 0.01',
          'learning_rate = 0.001',
          'learning_rate = 0.0001',
          'learning_rate = 0.00001',
          'learning_rate = 0.000001']

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'}]
             # {'c': 'blue', 'linestyle': '--'},
             # {'c': 'black', 'linestyle': '-'}
def plot_on_dataset(X, y, ax, ax2, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
#     ax.set_title(name)

    X = MinMaxScaler().fit_transform(X)
    mlps = []
    if name == "digits":
        # digits is larger but converges fairly quickly
        max_iter = 15
    else:
        max_iter = 10000

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(random_state=0,
                            max_iter=max_iter, **param)

        # some parameter combinations will not converge as can be seen on the
        # plots so they are ignored here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
            mlp.fit(X, y)

        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)

        y_pred = mlp.predict(X_test)
        F1 = ax2.plot(mlp.learning_rate, f1_score(y_test, y_pred,average='macro'))

    for mlp, label, args in zip(mlps, labels, plot_args):
        ax.plot(mlp.loss_curve_, label=label, **args)


fig, axes = plt.subplots(2, 1, figsize=(5, 5))
# load / generate some toy datasets

plot_on_dataset(X_train, y_train, ax=axes[0],ax2=axes[1], name='Games played')

fig.legend(axes[0].get_lines(), labels, loc=5)
plt.ylabel('Training set loss')
plt.xlabel('Number of epocs')
plt.title('Training set loss for differnt learning strategies')
plt.show()

