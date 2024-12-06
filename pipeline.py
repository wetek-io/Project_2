from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

def create_fit_encoders(X, columns):
    encoders = {}
    for col in X.columns:
        if col in columns['custom_categories']:
            encoders[col] = OrdinalEncoder(categories=[columns['custom_categories'][col]], handle_unknown='use_encoded_value', unknown_value=-1).fit(X=X[[col]])
        elif col in columns['yes_no_cols']:
            encoders[col] = OrdinalEncoder(categories=[["No", "Yes"]], handle_unknown='use_encoded_value', unknown_value=-1).fit(X=X[[col]])
        elif col in columns['label_cols']:
            encoders[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(X=X[[col]])
    return encoders

# def fit_encoders(X, encoders):
#     """Fit each encoder on its respective column."""
#     for col, encoder in encoders.items():
#         encoder.fit(X[[col]])  # Fit on the single column

def create_fit_scalers(X, columns):
    """Create and fit data to for the scaling columns."""
    scalers = {}
    for col in X.columns:
        if col in columns['min_max_cols']:
            scalers[col] = MinMaxScaler().fit(X[[col]])
        if col in columns['std_cols']:
            scalers[col] = StandardScaler().fit(X[[col]])
    return scalers

def transform(X, encoders, scalers):
    "Transform the data"
    X_transformed = X.copy()
    for col, encoder in encoders.items():
        X_transformed[col] = encoder.transform(X_transformed[[col]])
    for col, scaler in scalers.items():
        X_transformed[col] = scaler.transform(X_transformed[[col]])
    return X_transformed

def build_custom_cols(df):
    df["HeartFailureLikelihood"] = ((df['HadHeartAttack'] == "Yes") | (df["HadAngina"] == 'Yes')).astype(int)


def model_generator(df):
    custom_categories = {
        "Sex" : ["Male", "Female"],
        "AgeCategory" : ['Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 
                         'Age 35 to 39', 'Age 40 to 44', 'Age 45 to 49', 
                         'Age 50 to 54', 'Age 55 to 59', 'Age 60 to 64', 
                         'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79',
                         'Age 80 or older'],
        "HadDiabetes" : ['No', 'No, pre-diabetes or borderline diabetes', 
                         'Yes, but only during pregnancy (female)', 'Yes'],
        "GeneralHealth" : ["Poor", "Fair", "Good", "Very good", "Excellent"],
        "CovidPos" : ["No","Yes", 'Tested positive using home test without a health professional'],
        "LastCheckupTime" : ['Within past year (anytime less than 12 months ago)',
                             'Within past 2 years (1 year but less than 2 years ago)',
                             'Within past 5 years (2 years but less than 5 years ago)',
                             '5 or more years ago'],
        "RemovedTeeth" : ['None of them', '1 to 5', '6 or more, but not all', 'All'],
        "SmokerStatus" : ['Never smoked', 'Former smoker', 'Current smoker - now smokes some days', 
                          'Current smoker - now smokes every day'],
        "ECigaretteUsage" : ['Never used e-cigarettes in my entire life', 'Not at all (right now)', 
                             'Use them some days', 'Use them every day'],
        "TetanusLast10Tdap" : ['No, did not receive any tetanus shot in the past 10 years',
                               'Yes, received tetanus shot but not sure what type',
                               'Yes, received tetanus shot, but not Tdap', 'Yes, received Tdap'],        
    }

    yes_no_cols  = ['PhysicalActivities', 'HadStroke',
               'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder',
               'HadKidneyDisease', 'HadArthritis', 'DeafOrHardOfHearing',
               'BlindOrVisionDifficulty', 'DifficultyConcentrating',
               'DifficultyWalking', 'DifficultyDressingBathing',
               'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers',
               'HIVTesting','FluVaxLast12', 'PneumoVaxEver','HighRiskLastYear']
    label_cols   = ["RaceEthnicityCategory", "State"]
    min_max_cols = ["WeightInKilograms"]
    std_cols     = ["HeightInMeters", "SleepHours", "BMI", "PhysicalHealthDays", "MentalHealthDays"] # There ones are meant to be inverted

    columns = {
        'custom_categories' : custom_categories,
        "yes_no_cols" : yes_no_cols,
        'label_cols' : label_cols,
        'min_max_cols' : min_max_cols,
        'std_cols' : std_cols,
    }
    build_custom_cols(df)
    models = []

    print(" ----------------- NA REMOVED ----------------- ")
############################################################################################################################################################

    print("Run all columns\n")

    df_na_removed = df.dropna()
    X = df_na_removed.copy().drop(columns=["HeartFailureLikelihood", "HadHeartAttack", "HadAngina"])
    y = df_na_removed["HeartFailureLikelihood"]#.values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    encoders = create_fit_encoders(X_train, columns)
    scalers  = create_fit_scalers(X_train, columns)

    X_train_transformed = transform(X_train, encoders=encoders, scalers=scalers)
    X_test_transformed = transform(X_test, encoders=encoders, scalers=scalers)

    model = RandomForestClassifier(random_state=1, max_depth=7, n_estimators=256, class_weight="balanced")
    model.fit(X_train_transformed, y_train)
    models.append(model)

    y_train_pred = model.predict(X_train_transformed)
    y_test_pred = model.predict(X_test_transformed)

    print("Accuracy:")
    print(f"Score for train: {model.score(X_train_transformed, y_train):.4f}")
    print(f"Score for test: {model.score(X_test_transformed, y_test):.4f}")

    print("Recall:")
    print(f"Train recall: {recall_score(y_train, y_train_pred):.4f}")
    print(f"Test recall: {recall_score(y_test, y_test_pred):.4f}")

############################################################################################################################################################

    print("---------------------------------")
    print("With only the ones from Keepers:\n")
    keepers = ["Sex", "GeneralHealth", "PhysicalHealthDays", "SleepHours",
               "SmokerStatus", "ECigaretteUsage", "RaceEthnicityCategory", 
               "AgeCategory", "WeightInKilograms", "BMI", "AlcoholDrinkers",
               "HighRiskLastYear", "HeartFailureLikelihood"]

    X = df_na_removed.copy()[keepers]
    y = df_na_removed["HeartFailureLikelihood"]#.values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    encoders = create_fit_encoders(X_train, columns)
    scalers  = create_fit_scalers(X_train, columns)

    X_train_transformed = transform(X_train, encoders=encoders, scalers=scalers)
    X_test_transformed = transform(X_test, encoders=encoders, scalers=scalers)

    model = RandomForestClassifier(random_state=1, max_depth=7, n_estimators=256, class_weight="balanced")
    model.fit(X_train_transformed, y_train)
    models.append(model)

    y_train_pred = model.predict(X_train_transformed)
    y_test_pred = model.predict(X_test_transformed)

    print("Accuracy:")
    print(f"Score for train: {model.score(X_train_transformed, y_train):.4f}")
    print(f"Score for test: {model.score(X_test_transformed, y_test):.4f}")

    print("Recall:")
    print(f"Train recall: {recall_score(y_train, y_train_pred):.4f}")
    print(f"Test recall: {recall_score(y_test, y_test_pred):.4f}")

############################################################################################################################################################


    print("---------------------------------")
    print("With only the ones from the yes/no columns:\n")
    keepers = yes_no_cols + ["HeartFailureLikelihood"]

    X = df_na_removed.copy()[keepers]
    y = df_na_removed["HeartFailureLikelihood"]#.values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    encoders = create_fit_encoders(X_train, columns)
    scalers  = create_fit_scalers(X_train, columns)

    X_train_transformed = transform(X_train, encoders=encoders, scalers=scalers)
    X_test_transformed = transform(X_test, encoders=encoders, scalers=scalers)

    model = RandomForestClassifier(random_state=1, max_depth=7, n_estimators=256, class_weight="balanced")
    model.fit(X_train_transformed, y_train)
    models.append(model)

    y_train_pred = model.predict(X_train_transformed)
    y_test_pred = model.predict(X_test_transformed)

    print("Accuracy:")
    print(f"Score for train: {model.score(X_train_transformed, y_train):.4f}")
    print(f"Score for test: {model.score(X_test_transformed, y_test):.4f}")

    print("Recall:")
    print(f"Train recall: {recall_score(y_train, y_train_pred):.4f}")
    print(f"Test recall: {recall_score(y_test, y_test_pred):.4f}")

############################################################################################################################################################

    return models