# Project_2: Healthcare Outcome Prediction with Machine Learning

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Data Sources](#data-sources)
- [Quickstart Guide](#quickstart-guide)
- [Tech Stack Setup](#tech-stack-setup)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Interactive Map Features](#interactive-map-features)
- [Results and Insights](#results-and-insights)
- [Conclusion](#conclusion)
- [How to Use](#how-to-use)
- [References and Credits](#references-and-credits)
- [Deployment Workflow](#deployment-workflow)
- [Branch Discipline](#branch-discipline)
- [Team Members](#team-members)
- [Contact The Team](#contact-the-team)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

The primary goal of this project is to create a machine learning model that predicts healthcare outcomes such as heart disease, diabetes, or cancer risk. The focus is on ensuring: - Explainability: Understanding why the model makes specific predictions. - Fairness: Avoiding bias across demographic groups. - Effectiveness: Achieving high accuracy and actionable insights.

This model aims to assist healthcare providers in making informed decisions while promoting ethical and equitable AI usage.

## Objectives

We wanted to use machine learning to predict the likely hood that a patient has heart disease. To do this
we needed to be able to take in new patent data through a front-end webapp interface, then use that data to test against the model data to predict the with chances that the users currently has or will have heart disease

## Data-sources

The original data source was hosted on [Kraggle](https://www.kraggle.com), however due to the limitations of that specific dataset

## Quickstart Guide
1. Review and run the sections of main.ipynb to understand and perform the data extraction, cleaning, and transformation process. The cleaned data is exported as CSV files for the machine learning models.
2. Review and run the sections of model_dev.ipynb to initializes, train, and evaluate several models and their performance in the context of heart failure prediction.
3. Follow the Deployment Workflow guidelines below to deploy the gradio_app to huggingface.co, which gives you an easy-to-use web-based user interface.

## Tech Stack Setup
1. Clone the repository
2. conda env update --name project_2_env -f environment.yml --prune
3. Review and run main.ipynb
4. Review and run model_dev.ipynb
5. Deploy the gradio app following the Deployment Workflow steps below

## Exploratory Data Analysis (EDA)
Review and run the sections of model_dev.ipynb to initializes, train, and evaluate several models and their performance in the context of heart failure prediction.

## Interactive Map Features

## Results and Insights

## Conclusion

## How to Use

## References and Credits

## Deployment Workflow

The heartfailure prediction model was deployed to huggingface.co using a free account. Deployment to huggingface is easy. You simply sign up with an account, create a new [space](https://huggingface.co/docs/hub/en/spaces-overview), and then use git to push your model, python app, and any other dependencies. The app and backend dependencies will build automatically, and a URL will be provided.

In the case of this project, the files to be deployed are in the gradio_app folder.
1. app.py uses the [gradio](https://www.gradio.app) python library to build the interface.
2. requirements.txt informs the huggingface environment what backend requirements need to be installed
3. tuned_model.pkl is the AI model used to predict heart failure
4. features_used_in_model.csv allows the app to dynamically expand the features used, to a list. This makes it more efficient to code the interface, and to refactor everything if model changes are needed later.

## Branch Discipline

## Team Members

## Contact The Team

## Acknowledgments

### Data Sets

We will utilize healthcare datasets suitable for classification tasks, including:

### Setup

Build project environment using conda

```bash
conda env create -f environment.yml
```

**To remove the env, after installing a new tool and adding it to the yml**

```bash
conda remove --name project_2_env --all
```

### Tech Stack

[Environment](environment.yml)
