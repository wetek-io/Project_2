"""Quick and easy cleaning method for the dataset"""

import pandas as pd

from data.value_maps import (
    age_ranges,
    binaryValues,
    e_smoking_history,
    gen_health_weights,
    last_checkup,
    race_ethnicity_category,
    smoking_history,
    states,
)


def clean_data(df, path, fillna: bool = False, dropna: bool = True):
    """
    Simply takes in the dataframe from main.ipynb and cleans it up
    """

    # save the raw csv file
    pd.DataFrame.to_csv(df, f"{path}/raw_data.csv", index=False)
    
    # remap values to numerics
    df["GeneralHealth"] = df["GeneralHealth"].map(gen_health_weights)
    df["AgeCategory"] = df["AgeCategory"].map(age_ranges)
    df["RaceEthnicityCategory"] = df["RaceEthnicityCategory"].map(
        race_ethnicity_category
    )
    df["SmokerStatus"] = df["SmokerStatus"].map(smoking_history)
    df["ECigaretteUsage"] = df["ECigaretteUsage"].map(e_smoking_history)
    df["LastCheckupTime"] = df["LastCheckupTime"].map(last_checkup)
    df["State"] = df["State"].map(states)
    df["Sex"] = df["Sex"].map(binaryValues)
    df["PhysicalActivities"] = df["PhysicalActivities"].map(binaryValues)
    df["HadHeartAttack"] = df["HadHeartAttack"].map(binaryValues)
    df["HadAngina"] = df["HadAngina"].map(binaryValues)
    df["HadStroke"] = df["HadStroke"].map(binaryValues)
    df["HadArthritis"] = df["HadArthritis"].map(binaryValues)
    df["HadDiabetes"] = df["HadDiabetes"].map(binaryValues)
    df["AlcoholDrinkers"] = df["AlcoholDrinkers"].map(binaryValues)
    df["HighRiskLastYear"] = df["HighRiskLastYear"].map(binaryValues)
    
    # if you need to keep the na values
    if fillna:
        df = df.fillna(
            {
                "HighRiskLastYear": 9999,
                "AlcoholDrinkers": 9999,
                "AgeCategory": 9999,
                "RaceEthnicityCategory": 9999,
                "ECigaretteUsage": 9999,
                "SmokerStatus": 9999,
                "HadDiabetes": 9999,
                "HadArthritis": 9999,
                "HadKidneyDisease": 9999,
                "HadDepressiveDisorder": 9999,
                "HadAsthma": 9999,
                "HadStroke": 9999,
                "HadAngina": 9999,
                "HadHeartAttack": 9999,
                "PhysicalActivities": 9999,
                "LastCheckupTime": 9999,
                "GeneralHealth": 9999,
                "Sex": 9999,
                "PhysicalHealthDays": 9999,
                "MentalHealthDays": 9999,
                "SleepHours": 9999,
                "HeightInMeters": 9999,
                "BMI": 9999,
                "WeightInKilograms": 9999,
            }
        )

    if dropna:
        df = df.dropna(how='any')
        
    df = df.drop(columns=df.select_dtypes(include=["object"]).columns).reindex()
    pd.DataFrame.to_csv(df, f"{path}/clean_data.csv", index=False)
    return df
