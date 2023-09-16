import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data_metrics.csv')
df.head()
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()
print(confusion_matrix(df.actual_label.values, df.predicted_RF.values))
print(confusion_matrix(df.actual_label.values, df.predicted_LR.values))


def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))


def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))


def find_conf_matrix_values(y_true, y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN


def kormysh_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


print(kormysh_confusion_matrix(df.actual_label.values, df.predicted_RF.values))
print(kormysh_confusion_matrix(df.actual_label.values, df.predicted_LR.values))
assert np.array_equal(kormysh_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                      confusion_matrix(df.actual_label.values,
                                       df.predicted_RF.values)), 'svistel-nyk_confusion_matrix() is not correct for RF'
assert np.array_equal(kormysh_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                      confusion_matrix(df.actual_label.values,
                                       df.predicted_LR.values)), 'svistel-nyk_confusion_matrix() is not correct for LR'

print(accuracy_score(df.actual_label.values, df.predicted_RF.values))
print(accuracy_score(df.actual_label.values, df.predicted_LR.values))


def kormysh_accuracy_score(y_true, y_pred):
    # calculates the fraction of samples
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)


assert kormysh_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(
    df.actual_label.values,
    df.predicted_RF.values), 'kormysh_accuracy_score failed on assert kormysh_accuracy_score(df.actual_label.values, ' \
                             'df.predicted_LR.values) == accura-cy_score(df.actual_label.values, df.predicted_LR.values),' \
                             'kormysh_accuracy_score failed on LR'
print('Accuracy RF: % .3f' % (kormysh_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: % .3f' % (kormysh_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

print(recall_score(df.actual_label.values, df.predicted_RF.values))
print(recall_score(df.actual_label.values, df.predicted_LR.values))


def kormysh_recall_score(y_true, y_pred):
    # calculates the fraction of positive samples predicted correctly
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)


assert kormysh_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values,
                                                                                               df.predicted_RF.values), \
    'kormysh_recall_score failed on RF'
assert kormysh_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values,
                                                                                               df.predicted_LR.values), \
    'kormysh_recall_score failed on LR'
print('Recall RF: %.3f' % (kormysh_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f' % (kormysh_recall_score(df.actual_label.values, df.predicted_LR.values)))

print(precision_score(df.actual_label.values, df.predicted_RF.values))
print(precision_score(df.actual_label.values, df.predicted_LR.values))


def kormysh_precision_score(y_true, y_pred):
    # calculates the fraction of predicted positives samples that are actu-ally positive
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)


assert kormysh_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(
    df.actual_label.values, df.predicted_RF.values), 'svistel-nyk_precision_score failed on RF'
assert kormysh_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(
    df.actual_label.values, df.predicted_LR.values), 'svistel-nyk_precision_score failed on LR'
print('Precision RF: %.3f' % (kormysh_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f' % (kormysh_precision_score(df.actual_label.values, df.predicted_LR.values)))

print(f1_score(df.actual_label.values, df.predicted_RF.values))
print(f1_score(df.actual_label.values, df.predicted_LR.values))


def kormysh_f1_score(y_true, y_pred):
    # calculates the F1 score
    recall = kormysh_recall_score(y_true, y_pred)
    precision = kormysh_precision_score(y_true, y_pred)
    return (2 * (precision * recall)) / (precision + recall)


assert kormysh_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values,
                                                                                       df.predicted_RF.values), 'kormysh_f1_score failed on RF'
assert kormysh_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values,
                                                                                       df.predicted_LR.values), 'kormysh_f1_score failed on LR'
print('F1 RF: %.3f' % (kormysh_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (kormysh_f1_score(df.actual_label.values, df.predicted_LR.values)))

print('scores with threshold = 0.5')
print('Accuracy RF: %.3f' % (kormysh_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (kormysh_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (kormysh_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (kormysh_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f' % (
    kormysh_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f' % (kormysh_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f' % (
    kormysh_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f' % (kormysh_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
