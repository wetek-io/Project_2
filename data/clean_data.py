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
    fill_values,
)


def clean_data(df, path, fillna: bool = True):
    """
    Simply takes in the dataframe from main.ipynb and cleans it up
    """
    pd.DataFrame.to_csv(df, f"{path}/raw_data.csv", index=False)

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

    df = df.drop(columns=df.select_dtypes(include=["object"]).columns).reindex()

    if fillna:
        df = df.fillna(value=fill_values)

    pd.DataFrame.to_csv(df, f"{path}/clean_data.csv", index=False)

    return df
