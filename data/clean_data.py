import pandas as pd
from data.value_maps import (
    e_smoking_history,
    last_checkup,
    smoking_history,
    race_ethnicity_category,
    age_ranges,
    gen_health_weights,
    states,
    binaryValues,
)


def clean_data(df, path):
    pd.DataFrame.to_csv(df, f"{path}/raw_data.csv")
    df = df.fillna(
        {
            "HighRiskLastYear": df["HighRiskLastYear"].mode()[0],
            "AlcoholDrinkers": df["AlcoholDrinkers"].mode()[0],
            "AgeCategory": df["AgeCategory"].mode()[0],
            "RaceEthnicityCategory": df["RaceEthnicityCategory"].mode()[0],
            "ECigaretteUsage": df["ECigaretteUsage"].mode()[0],
            "SmokerStatus": df["SmokerStatus"].mode()[0],
            "HadDiabetes": df["HadDiabetes"].mode()[0],
            "HadArthritis": df["HadArthritis"].mode()[0],
            "HadKidneyDisease": df["HadKidneyDisease"].mode()[0],
            "HadDepressiveDisorder": df["HadDepressiveDisorder"].mode()[0],
            "HadAsthma": df["HadAsthma"].mode()[0],
            "HadStroke": df["HadStroke"].mode()[0],
            "HadAngina": df["HadAngina"].mode()[0],
            "HadHeartAttack": df["HadHeartAttack"].mode()[0],
            "PhysicalActivities": df["PhysicalActivities"].mode()[0],
            "LastCheckupTime": df["LastCheckupTime"].mode()[0],
            "GeneralHealth": df["GeneralHealth"].mode()[0],
            "Sex": df["Sex"].mode()[0],
            "SleepHours": df["SleepHours"].median(),
            "HeightInMeters": df["HeightInMeters"].median(),
            "BMI": df["BMI"].median(),
            "WeightInKilograms": df["WeightInKilograms"].median(),
            "State": "Unknown",
        }
    )
    df["GeneralHealthIDs"] = df["GeneralHealth"].map(gen_health_weights)
    df["AgeCategoryIDs"] = df["AgeCategory"].map(age_ranges)
    df["RaceEthnicityCategoryIDs"] = df["RaceEthnicityCategory"].map(
        race_ethnicity_category
    )
    df["SmokerStatusIDs"] = df["SmokerStatus"].map(smoking_history)
    df["ECigaretteUsageIDs"] = df["ECigaretteUsage"].map(e_smoking_history)
    df["LastCheckupTimeIDs"] = df["LastCheckupTime"].map(last_checkup)
    df["StateIDs"] = df["State"].map(states)
    df["SexBI"] = df["Sex"].map(binaryValues)
    df["PhysicalActivitiesBI"] = df["PhysicalActivities"].map(binaryValues)
    df["HadHeartAttackBI"] = df["HadHeartAttack"].map(binaryValues)
    df["HadAnginaBI"] = df["HadAngina"].map(binaryValues)
    df["HadStrokeBI"] = df["HadStroke"].map(binaryValues)
    df["HadArthritisBI"] = df["HadArthritis"].map(binaryValues)
    df["HadDiabetesBI"] = df["HadDiabetes"].map(binaryValues)
    df["AlcoholDrinkersBI"] = df["AlcoholDrinkers"].map(binaryValues)
    df["HighRiskLastYearBI"] = df["HighRiskLastYear"].map(binaryValues)

    df = df.drop(columns=df.select_dtypes(include=["object"]).columns)
    pd.DataFrame.to_csv(df, f"{path}/clean_data.csv")
    return df
