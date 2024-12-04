import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import GradientBoostingClassifier
df = pd.read_csv('data/data.csv')
print('Columns and their dtypes:')
print(df.dtypes)
print("\nUnique values in columns:")
heart_related_columns = [col for col in df.columns if 'Heart' in col or 'Cardiac' in col or 'Attack' in col]
for col in heart_related_columns:
    print(f"\n{col} unique values:")
    print(df[col].unique())
print(df.head())
df['HeartDisease'] = ((df['HadHeartAttackBI'] == 1 ) | (df['HadAnginaBI'] == 1)).astype(int)
print("\nHeartDisease column distribution:")
print(df['HeartDisease'].value_counts())
x = df.drop(['Unnamed: 0', 'HeartDisease'], axis=1)
y = df["HeartDisease"]                                                      
x = x.fillna(x.mean())
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=79, stratify=y)
print("\nTraining set class distribution:")
print(y_train.value_counts())
print("\nTesting set class distribution:")
print(y_test.value_counts())
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
y_train_encoded
scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_train_scaled
x_test_scaled = scaler.transform(x_test)
x_test_scaled
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced'), param_grid, cv=5)
grid_search.fit(x_train_scaled, y_train_encoded)
print("Best parameters:", grid_search.best_params_)
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)
lr_model.fit(x_train_scaled, y_train_encoded)
print(f"Training Data Score: {lr_model.score(x_train_scaled, y_train_encoded)}")
print(f"Testing Data Score: {lr_model.score(x_test_scaled, y_test_encoded)}")
scores = cross_val_score(lr_model, x_train_scaled, y_train_encoded, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean CV Score:", scores.mean())
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=5, min_samples_split=10)
rf_model.fit(x_train_scaled, y_train_encoded)
print(f"Training Data Score: {rf_model.score(x_train_scaled, y_train_encoded)}")
print(f"Testing Data Score: {rf_model.score(x_test_scaled, y_test_encoded)}")
dt_model = DecisionTreeClassifier(class_weight='balanced', max_depth=5, min_samples_split=10, min_samples_leaf=5)
dt_model.fit(x_train_scaled, y_train_encoded)
print(f"Training Data Score: {dt_model.score(x_train_scaled, y_train_encoded)}")
print(f"Testing Data Score: {dt_model.score(x_test_scaled, y_test_encoded)}")
gb_model = GradientBoostingClassifier(max_depth=3, n_estimators=100, learning_rate=0.1)
gb_model.fit(x_train_scaled, y_train_encoded)
print(f"Training Data Score: {gb_model.score(x_train_scaled, y_train_encoded)}")
print(f"Testing Data Score: {gb_model.score(x_test_scaled, y_test_encoded)}")


# import joblib
# joblib.dump(best_model, 'optimized_heart_failure')
# def build_logistic_regression_model(x, y, test_size=0.2, random_state=79):
#     x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = \
#     test_size, random_state = random_state, stratify = y)
#     scaler = StandardScaler()
#     x_train_scaled = scaler.fit_transform(x_train)
#     x_test_scaled = scaler.transform(x_test)
#     cv_scores ={'precision': cross_val_score(LogisticRegression\
#     (random_state=random_state),x_train_scaled, y_train, scoring='precision', \
#     cv=5),'recall': cross_val_score(LogisticRegression\
#     (random_state=random_state),x_train_scaled, y_train, \
#     scoring='recall', cv=5),'f1': cross_val_score\
#     (LogisticRegression(random_state=random_state),x_train_scaled,\
#     y_train, scoring='f1', cv=5)}
#     model =LogisticRegression(random_state=random_state, max_iter= 1000, \
#         class_weight="balanced")
#     model.fit(x_train_scaled, y_train)
#     y_pred = model.predict(x_test_scaled)
#     y_pred_proba = model.predict_proba(x_test_scaled)[:,1]
#     performance_metrics ={'precision': precision_score(y_test, y_pred),
#         'recall': recall_score(y_test, y_pred),
#         'f1_score': f1_score(y_test, y_pred),
#         'roc_auc': roc_auc_score(y_test, y_pred_proba),
#         'average_precision': average_precision_score(y_test, y_pred_proba),
#         'cross_val_precision': np.mean(cv_scores['precision']),
#         'cross_val_recall': np.mean(cv_scores['recall']),
#         'cross_val_f1': np.mean(cv_scores['f1'])}
#     precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
#     plt.figure(figsize=(10, 6))
#     plt.plot(recall, precision, label='Precision-Recall curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('precision_recall_curve.png')
#     plt.close()
#     fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#     plt.figure(figsize=(10, 6))
#     plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {performance_metrics["roc_auc"]:.2f})')
#     plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('roc_curve.png')
#     plt.close() 
#     return {'model': model,'scaler': scaler, 
#         'performance_metrics': performance_metrics,
#         'cross_validation_scores': cv_scores}
