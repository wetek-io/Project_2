import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ( precision_recall_curve, average_precision_score, \
    roc_auc_score, precision_score, recall_score, f1_score, roc_curve)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import GradientBoostingClassifier


df=pd.read_csv('data/data.csv')
df.head()
print(df.head())
# def build_logistic_regression_model(x, y, test_size=0.2, random_state=79):
#     x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = \
# test_size, random_state = random_state, stratify = y)
#     scaler = StandardScaler()
#     x_train_scaled = scaler.fit_transform(x_train)
#     x_test_scaled = scaler.transform(x_test)
# cv_scores ={'precision': cross_val_score(LogisticRegression\
#     (random_state=random_state),x_train_scaled, y_train, scoring='precision', \
#     cv=5),'recall': cross_val_score(LogisticRegression\
#     (random_state=random_state),x_train_scaled, y_train, \
#     scoring='recall', cv=5),'f1': cross_val_score\
#     (LogisticRegression(random_state=random_state),x_train_scaled,\
#     y_train, scoring='f1', cv=5)}
# model =LogisticRegression(random_state=random_state, max_iter= 1000, \
#         class_weight="balanced")
# model.fit(x_train_scaled, y_train)
# y_pred = model.predict(x_test_scaled)
# y_pred_proba = model.predict_proba(x_test_scaled)[:,1]
# performance_metrics ={'precision': precision_score(y_test, y_pred),
#         'recall': recall_score(y_test, y_pred),
#         'f1_score': f1_score(y_test, y_pred),
#         'roc_auc': roc_auc_score(y_test, y_pred_proba),
#         'average_precision': average_precision_score(y_test, y_pred_proba),
#         'cross_val_precision': np.mean(cv_scores['precision']),
#         'cross_val_recall': np.mean(cv_scores['recall']),
#         'cross_val_f1': np.mean(cv_scores['f1'])}
# precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
# plt.figure(figsize=(10, 6))
# plt.plot(recall, precision, label='Precision-Recall curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend()
# plt.tight_layout()
# plt.savefig('precision_recall_curve.png')
# plt.close()
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# plt.figure(figsize=(10, 6))
# plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {performance_metrics["roc_auc"]:.2f})')
# plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend()
# plt.tight_layout()
# plt.savefig('roc_curve.png')
# plt.close() 
# return {'model': model,'scaler': scaler, 
#         'performance_metrics': performance_metrics,
#         'cross_validation_scores': cv_scores}