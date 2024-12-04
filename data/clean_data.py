"""Quick and easy cleaning method for the dataset"""

from pathlib import Path
import pandas as pd
import logging

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

CSV_URL = "https://web-app-media-assests.sfo3.cdn.digitaloceanspaces.com/Indicators_of_Heart_Disease/2022/heart_2022_with_nans.csv"
RAW_DATA_PATH = "./data/csv/raw_data.csv"
CLEAN_DATA_PATH = "./data/csv/clean_data.csv"


def clean_data(df, path, fillna: bool = True, dropna: bool = False):
    """
    Cleans the input DataFrame by applying mappings, filling/dropping NaN values,
    and saving raw and cleaned versions of the data.

    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        path (str): Directory path to save raw and cleaned data.
        fillna (bool): Whether to fill missing values.
        dropna (bool): Whether to drop rows with missing values.
        target_columns (list[str], optional): List of columns to retain.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """

    # Initialize logger
    logging.basicConfig(level=logging.INFO)

    # Save raw data
    logging.info("Saving raw data...")

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

    if fillna:
        df = df.fillna(value=fill_values)
    if dropna:
        df = df.dropna(how="any")

    df = df.drop(columns=df.select_dtypes(include=["object"]).columns).reindex()

    pd.DataFrame.to_csv(df, f"{path}/clean_data.csv", index=False)

    return df


def fetch_check(
    to_fetch: bool = False, to_fillna: bool = False, to_dropna: bool = False
):
    """
    Check for fetch and data cleaning toggles
    """

    # check for existing file and not_fetch bool stats
    if not Path(CLEAN_DATA_PATH).exists() or to_fetch:
        df = pd.read_csv(CSV_URL)
        df.to_csv(RAW_DATA_PATH)
        df = clean_data(df, "data/csv/", fillna=to_fillna, dropna=to_dropna)
    else:
        df = pd.read_csv(CLEAN_DATA_PATH)

    return df
