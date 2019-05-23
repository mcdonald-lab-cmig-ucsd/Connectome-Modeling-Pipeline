from sklearn.preprocessing import *
from sklearn.decomposition import *
from sklearn.metrics import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def scale_data(train, test, scaler = StandardScaler()):
    """Scale the training and testing data.

    Scales the train/test datasets with the specified scaler.
    The scaler should conform to sklearn API (i.e. have fit_transform,
    transform, fit functions).

    Args:
        1. train       -- training data
        2. test        -- testing data
        3. scaler      -- scaler object with sklearn API functions (Default: StandardScaler)

    Returns:
        Scaled training and testing datasets.
    """

    train_sc = scaler.fit_transform(train)
    test_sc = scaler.transform(test)

    return train_sc,test_sc

def select_features(train, test, method = 'pca', **kwargs):
    """Select the relevant features

    Scales the train/test datasets with the specified scaler.
    The scaler should conform to sklearn API (i.e. have fit_transform,
    transform, fit functions).

    Args:
        1. train       -- training data
        2. test        -- testing data
        3. scaler      -- scaler object with sklearn API functions (Default: StandardScaler)

    Returns:
        Scaled training and testing datasets.
    """

    # TODO: maybe add graph theory measures to select features

    if method == 'pca': # pca feature selection
        return do_pca(train, test, **kwargs)
    return None,None,None


def do_pca(train, test, **kwargs):
    pca = PCA(**kwargs)
    train_tr = pca.fit_transform(train)
    test_tr = pca.transform(test)

    return pca,train_tr,test_tr


def print_stats(confusion_matrix, actually_print = True):
    tn, fp, fn, tp = confusion_matrix.ravel()

    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    if actually_print:
        print("Accuracy: {}".format(acc))
        print("PPV: {}".format(ppv))
        print("NPV: {}".format(npv))
        print("Sensitivity: {}".format(sensitivity))
        print("Specificity: {}".format(specificity))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1-score: {}".format(f1_score))
    else:
        return (acc, ppv, npv, sensitivity, specificity)

def plot_feature_importance(model, title, feature_names = None,
        top_n = 10, show = False, filename = None):
    if feature_names is None:
        feature_names = ['PC-{}'.format(i + 1) for
            i in range(len(model.feature_importances_))]
    # Calculate feature importances
    importances = model.feature_importances_

    if top_n == 0 or top_n > len(feature_names):
        top_n = len(feature_names)

    # write importance vector to csv file
    csv_filename = title.replace(' ', '_')
    print('Writing feature importances to csv file: {}.csv'.format(csv_filename))
    pd.DataFrame({'names':feature_names,
        'imp': importances}).to_csv('{}.csv'.format(csv_filename))

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    indices = indices[:top_n]

    indices = np.flip(indices)

    # Rearrange feature names so they match the sorted feature importances
    names = [feature_names[i] for i in indices]

    plt.figure(figsize=(9,8))
    # Barplot: Add bars
    plt.barh(range(top_n), importances[indices])
    # Add feature names as x-axis labels
    plt.yticks(range(top_n), names, fontsize = 10)

    # Create plot title
    plt.title(title)
    plt.xlabel("Relative Importance")

    if show: # Show plot
        plt.show()
    if filename: # save figure to png image
        if not filename.endswith('.png'):
            filename = '{}.png'.format(filename)

        plt.savefig(filename)


def plot_auc(y_true, y_proba, title, show = False, filename = None):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    roc_class = 1

    plt.figure()
    lw = 2

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

    # chance (dumb classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    if show:
        plt.show()

    if filename: # save figure to png image
        if not filename.endswith('.png'):
            filename = '{}.png'.format(filename)

        plt.savefig(filename)
