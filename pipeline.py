from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

class CustomEncoders(BaseEstimator, TransformerMixin): #GPT inital generated
    def __init__(self, columns):  # Optionally accept predefined encoders
        self.encoders = {}
        self.custom_categories  = columns['custom_categories']
        self.yes_no_cols        = columns['yes_no_cols']
        self.label_cols         = columns['label_cols']

    def fit(self, X, y=None):
        """Fit encoders based on the data."""
        self.encoders = self.create_encoders(X)
        self.fit_encoders(X)
        return self
    
    def transform(self, X):
        "Transform the data"
        X_transformed = X.copy()
        for col, encoder in self.encoders.items():
            X_transformed[col] = encoder.transform(X_transformed[[col]])
        return X_transformed.values
    
    def create_encoders(self, X):
        encoders = {}
        for col in X.columns:
            if col in self.custom_categories:
                encoders[col] = OrdinalEncoder(categories=[self.custom_categories[col]], handle_unknown='use_encoded_value', unknown_value=-1)
            elif col in self.yes_no_cols:
                encoders[col] = OrdinalEncoder(categories=["No", "Yes"], handle_unknown='use_encoded_value', unknown_value=-1)
            elif col in self.label_cols:
                encoders[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        return encoders

    def fit_encoders(self, X):
        """Fit each encoder on its respective column."""
        for col, encoder in self.encoders.items():
            encoder.fit(X[[col]])  # Fit on the single column


class CustomScalers(BaseEstimator, TransformerMixin): #GPT inital generated
    def __init__(self, columns):  # Optionally accept predefined encoders
        self.scalers = {}
        self.min_max_cols       = columns['min_max_cols']
        self.std_cols           = columns['std_cols']   
        self.inv_std_cols       = columns['inv_std_cols']

    def fit(self, X, y=None):
        """Fit encoders based on the data."""
        self.scalers = self.create_fit_scalers(X)
        return self
    
    def transform(self, X):
        "Transform the data"
        X_transformed = X.copy()
        for col, scaler in self.scalers.items():
            X_transformed[col] = scaler.transform(X_transformed[[col]])
        return X_transformed.values

    def create_fit_scalers(self, X):
        """Create and fit data to for the scaling columns."""
        scalers = {}
        for col in X.columns:
            if col in self.min_max_cols:
                scalers[col] = MinMaxScaler().fit(X[[col]].values.reshape(-1,1))
            if col in self.std_cols:
                scalers[col] = StandardScaler().fit(X[[col]].values.reshape(-1,1))
            if col in self.inv_std_cols:
                scalers[f"{col}_inverted"] = StandardScaler().fit(np.log1p((X[[col]].max() - X[[col]])).values.reshape(-1,1))


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

    yes_no_cols = ['PhysicalActivities', 'HadStroke',
               'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder',
               'HadKidneyDisease', 'HadArthritis', 'DeafOrHardOfHearing',
               'BlindOrVisionDifficulty', 'DifficultyConcentrating',
               'DifficultyWalking', 'DifficultyDressingBathing',
               'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers',
               'HIVTesting','FluVaxLast12', 'PneumoVaxEver','HighRiskLastYear']
    
    label_cols   = ["RaceEthnicityCategory", "State"]
    min_max_cols = ["WeightInKilograms"]
    std_cols     = ["HeightInMeters", "SleepHours", "BMI"]
    inv_std_cols = ["PhysicalHealthDays", "MentalHealthDays"] # There ones are meant to be inverted

    categorical_columns = {
        'custom_categories' : custom_categories,
        'yes_no_cols' : yes_no_cols,
        'label_cols' : label_cols,
        }
    
    numerical_columns = {
        'min_max_cols' : min_max_cols,
        'std_cols' : std_cols,
        'inv_std_cols' : inv_std_cols,
        }


    categorical_transformer = CustomEncoders(columns=categorical_columns)
    numerical_transformer = CustomScalers(columns=numerical_columns)

    categorical_features = list(custom_categories.keys()) + yes_no_cols + label_cols
    numerical_features = min_max_cols + std_cols + inv_std_cols

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features),
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocesspr", preprocessor),
        ('model', RandomForestClassifier(random_state=1))
    ])

    X_train, X_test, y_train, t_test = train_test_split(X, y, random_state=1)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(y_pred)

if __name__ == "__main__":
    heart_2022_df = pd.read_csv("heart_2022_no_nans.csv")
    heart_2022_df["HeartFailureLikelihood"] = ((heart_2022_df['HadHeartAttack'] == "Yes") | (heart_2022_df["HadAngina"] == 'Yes')).astype(int)
    X = heart_2022_df.copy().drop(columns=["HeartFailureLikelihood", "HadHeartAttack", "HadAngina"])
    y = heart_2022_df["HeartFailureLikelihood"]#.values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model_generator(heart_2022_df)