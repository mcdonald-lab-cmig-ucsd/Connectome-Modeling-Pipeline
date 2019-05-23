#!/bin/python3

from connectome_predictions import ConnectomeModel
from utils import *
from sklearn.metrics import roc_curve
from xgboost import *

cm = ConnectomeModel('../connectomes/', '../data_np/')
labels,subjectlist = cm.read_impair('final_final_cog_conn.csv', 'LangImp')

data = cm.read_data('connectome')
cm.split_data() # default ucsd/ucsf split

X_train = data[cm.train]
y_train = labels[cm.train]
X_test = data[cm.test]
y_test = labels[cm.test]

X_train,X_test = scale_data(X_train, X_test)

print(X_train.shape)
print(X_test.shape)


## DEFINE parameters for pca and xgboost ##

pca_params = {
        'n_components': 40,
        'random_state': 2
}

xgb_params = {
    'verbosity': 3,
    'max_depth': 10,
    'n_estimators': 1000,
    'scale_pos_weight': 1.4,
    'min_child_weight': 6.5
}

##

# feature selection here (pca in this case)
pca,X_train,X_test = select_features(X_train,
        X_test,
        method = 'pca',
        **pca_params)

#good_pcs = [2, 20, 39, 11]
good_pcs = np.arange(0, 40)
X_train = X_train[:,good_pcs]
X_test = X_test[:,good_pcs]

# create model
model = XGBClassifier(**xgb_params)

# train and test
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)

# find best threshold based on roc curve
best_thr = cm.find_best_threshold(y_test, y_proba[:,1])
y_pred = y_proba[:,1] > best_thr

# print out a confusion table and stats
conf_matr = confusion_matrix(y_true = y_test, y_pred = y_pred)

print(conf_matr)
print_stats(conf_matr)

# plot the roc curve
plot_auc(y_test, y_proba[:,1], 'Connectome Language', False, 'connectome_language.png')

# plot feature importance
feature_names = ['PC-{}'.format(i + 1) for i in good_pcs]
plot_feature_importance(model, "Feature importance for connectome", feature_names, 10,
        False, 'feature_importance_connectome_language.png')

# randomization testing
perf_distribution = \
        cm.randomization_test(X_train, y_train, X_test, y_test, 1000, **xgb_params)

print(perf_distribution.describe())
