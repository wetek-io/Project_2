import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def load_and_preprocess_data(filepath):

    df = pd.read_csv('data\csv\clean_data.csv')
    print('Columns and their dtypes:')
    print(df.dtypes)

    heart_related_columns = [col for col in df.columns if 'Heart' in col or 'Cardiac' in col or 'Attack' in col]
    for col in heart_related_columns:
        print(f"\n{col} unique values:")
        print(df[col].unique())

    df['HeartDisease'] = ((df['HadHeartAttack'] == 1 ) | (df['HadAngina'] == 1)).astype(int)
    print("\nHeartDisease column distribution:")
    print(df['HeartDisease'].value_counts())

    x = df.drop(['HeartDisease'], axis=1)
    y = df["HeartDisease"]        

    x = x.fillna(x.mean())

    return x,y 

def split_and_scale_data(x,y, random_state=79):

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, stratify=y)

    print("\nTraining set class distribution:")
    print(y_train.value_counts())
    print("\nTesting set class distribution:")
    print(y_test.value_counts())

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train_encoded, y_test_encoded

def train_and_evaluate_models(x_train, x_test, y_train, y_test):

    models = {'Logistic Regression': (
            LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1),
            {}
        ),
        'Random Forest': (
            RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced', 
                max_depth=5, 
                min_samples_split=10
            ),
            {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        ),
        'Decision Tree': (
            DecisionTreeClassifier(
                class_weight='balanced', 
                max_depth=5, 
                min_samples_split=10, 
                min_samples_leaf=5
            ),
            {}
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(
                max_depth=3, 
                n_estimators=100, 
                learning_rate=0.1
            ),
            {}
        )
    }
    trained_models = {}
    for name, (model, param_grid) in models.items():
        print(f"\n--- {name} ---")
    
        if param_grid: 
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(x_train, y_train)
            model = grid_search.best_estimator_
            print("Best parameters:", grid_search.best_params_)

        model.fit(x_train, y_train)

        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)

        print(f"Training Data Score: {train_score}")
        print(f"Testing Data Score: {test_score}")

        if name == 'Logistic Regression':
            cv_scores = cross_val_score(model, x_train, y_train, cv=5)
            print("Cross-Validation Scores:", cv_scores)
            print("Mean CV Score:", cv_scores.mean())

        trained_models[name] = model

    return trained_models

def main():
    
    filepath = 'data/csv/clean_data.csv'

    x, y, = load_and_preprocess_data(filepath)

    x_train_scaled, x_test_scaled, y_train_encoded, y_test_encoded = split_and_scale_data(x, y)

    trained_models = train_and_evaluate_models(
        x_train_scaled, x_test_scaled, 
        y_train_encoded, y_test_encoded)
    
if __name__ == "__main__":
    main()

