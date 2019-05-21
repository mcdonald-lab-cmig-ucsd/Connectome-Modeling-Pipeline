from sklearn.preprocessing import *
from sklearn.decomposition import *

def scale_data(train, test, scaler = StandardScaler()):
    train_sc = scaler.fit_transform(train)
    test_sc = scaler.transform(test)

    return train_sc,test_sc

def select_features(train, test, method = 'pca', **kwargs):
    if method == 'pca':
        return do_pca(train, test, **kwargs)


def do_pca(train, test, **kwargs):
    pca = PCA(**kwargs)
    train_tr = pca.fit_transform(train)
    test_tr = pca.transform(test)

    return pca,train_tr,test_tr
