"""Quick and easy cleaning method for the dataset"""

import logging
from pathlib import Path

import pandas as pd

from data.value_maps import category_maps, binary_maps, fill_values

CSV_URL = "https://web-app-media-assests.sfo3.cdn.digitaloceanspaces.com/Indicators_of_Heart_Disease/2022/heart_2022_with_nans.csv"
RAW_DATA_PATH = "./data/csv/raw_data.csv"
CLEAN_DATA_PATH = "./data/csv/clean_data.csv"


def clean_data(
    df,
    path,
    fillna: bool = False,
    dropna: bool = False,
    remap_values: bool = True,
    drop_obj: bool = True,
):
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

    if remap_values:
        df["GeneralHealth"] = df["GeneralHealth"].map(category_maps["GeneralHealth"])
        df["AgeCategory"] = df["AgeCategory"].map(category_maps["AgeCategory"])
        df["RaceEthnicityCategory"] = df["RaceEthnicityCategory"].map(
            category_maps["RaceEthnicityCategory"]
        )
        df["SmokerStatus"] = df["SmokerStatus"].map(category_maps["SmokerStatus"])
        df["ECigaretteUsage"] = df["ECigaretteUsage"].map(
            category_maps["ECigaretteUsage"]
        )
        df["LastCheckupTime"] = df["LastCheckupTime"].map(
            category_maps["LastCheckupTime"]
        )
        df["State"] = df["State"].map(category_maps["States"])
        df["Sex"] = df["Sex"].map(binary_maps["Sex"])
        df["PhysicalActivities"] = df["PhysicalActivities"].map(
            binary_maps["PhysicalActivities"]
        )
        df["HadHeartAttack"] = df["HadHeartAttack"].map(binary_maps["HadHeartAttack"])
        df["HadAngina"] = df["HadAngina"].map(binary_maps["HadAngina"])
        df["HadStroke"] = df["HadStroke"].map(binary_maps["HadStroke"])
        df["HadArthritis"] = df["HadArthritis"].map(binary_maps["HadArthritis"])
        df["HadDiabetes"] = df["HadDiabetes"].map(binary_maps["HadDiabetes"])
        df["AlcoholDrinkers"] = df["AlcoholDrinkers"].map(
            binary_maps["AlcoholDrinkers"]
        )
        df["HighRiskLastYear"] = df["HighRiskLastYear"].map(
            binary_maps["HighRiskLastYear"]
        )
    if fillna:
        df = df.fillna(value=fill_values)
    if dropna:
        df = df.dropna(how="any")
    if drop_obj:
        df = df.drop(columns=df.select_dtypes(include=["object"]).columns).reindex()

    df = df.drop_duplicates(
        subset=None, keep="first", inplace=False, ignore_index=False
    )

    pd.DataFrame.to_csv(df, f"{path}/clean_data.csv", index=False)

    return pd.DataFrame(df)


def fetch_check(
    to_fetch: bool = False,
    to_fillna: bool = False,
    to_dropna: bool = False,
    to_remap: bool = True,
    drop_obj: bool = True,
):
    """
    Check for fetch and data cleaning toggles
    """

    # check for existing file and not_fetch bool stats
    if not Path(CLEAN_DATA_PATH).exists() or to_fetch:
        df = pd.read_csv(CSV_URL)
        df.to_csv(RAW_DATA_PATH)
        df = clean_data(
            df,
            "data/csv/",
            fillna=to_fillna,
            dropna=to_dropna,
            remap_values=to_remap,
            drop_obj=drop_obj,
        )
    else:
        df = pd.read_csv(CLEAN_DATA_PATH)

    return df
