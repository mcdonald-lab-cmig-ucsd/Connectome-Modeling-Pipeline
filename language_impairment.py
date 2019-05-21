#!/bin/python3

from connectome_predictions import ConnectomeModel
from utils import *

cm = ConnectomeModel('../../connectomes/', '../../data_np/')
labels,subjectlist = cm.read_impair('final_final_cog_conn.csv', 'LangImp')

data = cm.read_data('connectome')
cm.split_data()

X_train = data[cm.train]
y_train = labels[cm.train]
X_test = data[cm.test]
y_test = labels[cm.test]

X_train,X_test = scale_data(X_train, X_test)

pca,X_train,X_test = select_features(X_train, X_test, method = 'pca', **{'n_components': 40, 'random_state': 2})

print(X_train.shape)
print(X_test.shape)
