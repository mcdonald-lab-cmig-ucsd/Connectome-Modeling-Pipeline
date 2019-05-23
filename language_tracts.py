#!/bin/python3

from connectome_predictions import ConnectomeModel
from utils import *
from sklearn.metrics import roc_curve
from xgboost import *
import numpy as np

cm = ConnectomeModel('../connectomes/', '../data_np/')
labels,subjectlist = cm.read_impair('final_final_cog_conn.csv', 'LangImp')

data = cm.read_data('tract', tract_file = 'language_tracts.csv',
        tract_regex = '_FA')
cm.split_data() # default ucsd/ucsf split

X_train = data[cm.train]
y_train = labels[cm.train]
X_test = data[cm.test]
y_test = labels[cm.test]

X_train,X_test = scale_data(X_train, X_test)

print(X_train.shape)
print(X_test.shape)


## DEFINE parameters for xgboost ##
xgb_params = {
    'verbosity': 3,
    'n_estimators': 200,
    'scale_pos_weight': 1.1,
    'min_child_weight': 2,
    'colsample_bytree': 0.3
}

##

# create model
model = XGBClassifier(**xgb_params)

# train and test
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)

# find best threshold based on roc curve
best_thr = cm.find_best_threshold(y_test, y_proba[:,1])
y_pred = y_proba[:,1] > best_thr

#print out a confusion table and stats
conf_matr = confusion_matrix(y_true = y_test, y_pred = y_pred)

print(conf_matr)
print_stats(conf_matr)

# plot the roc curve
plot_auc(y_test, y_proba[:,1], 'Tracts Language', False, 'tracts_language.png')

# plot feature importance
plot_feature_importance(model, "Feature importance for tracts", cm.feature_names, 0,
        False, 'feature_importance_tract_language.png')
