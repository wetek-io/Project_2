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

## Executive Summary
The goal of this project was to develop a transparent, fair, and effective machine learning model for predicting heart disease risk. By refining a dataset originally sourced from Kaggle, we aimed to equip healthcare providers with a data-driven tool that prioritizes ethical and equitable decision-making. Throughout the workflow, data was extracted, cleaned, and transformed using `main.ipynb`, and multiple classification models were trained, evaluated, and threshold-tuned in `model_dev.ipynb` to maximize their diagnostic value. Central to this effort was the use of the ROC AUC metric, which is more appropriate than simple accuracy for assessing classification performance, especially when dealing with imbalanced data and high-stakes clinical outcomes. The final Random Forest model, after threshold tuning, achieved a respectable ROC AUC (~0.74) while prioritizing recall to minimize the risk of missing patients who are potentially at high risk. Although this approach sacrificed some precision, the decision was guided by the critical importance of reducing false negatives in a medical context. Additionally, other models, such as Logistic Regression, showed promise for similar performance if further threshold tuning were pursued. The completed solution has been deployed using Gradio on Hugging Face Spaces, enabling an accessible, web-based interface that healthcare professionals can easily use to input new patient data and receive predictions. Looking ahead, there is potential to improve fairness by exploring bias mitigation techniques, enhance model robustness through advanced feature engineering and hyperparameter optimization, and further refine metrics to ensure that the model’s predictive power directly translates into better patient outcomes. In essence, this project offers a strong foundation for an interpretable, equitable, and actionable heart disease risk prediction tool.  

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
With respect to model optimization, our primary focus was on reducing false negatives rather than model accuracy or R-squared. False negatives could be catastrophic, because you would be telling someone with a higher likelihood of heart failure, that they are not at risk of heart failure.\
Note that ROC AUC is the correct metric to evaluate our model’s performance, not R-squared, because the focus is on the Model’s ability to distinguish classes. It's a classificaiton problem.\
In contrast, R-squared focuses on variance explained in the target variable, for regression problems.

Advantages of ROC AUC
	•	Threshold-Independent:
	•	Unlike accuracy, it evaluates model performance across all classification thresholds.
	•	Class-Imbalance Robustness:
	•	It works well even with imbalanced datasets since it focuses on the ranking of predictions.

As you will see in model_dev.ipynb, these are the resulting scores from Threshold-tuning the random forest model. Notice that prioritizing recall has sacrificed precision.

Threshold-Tuned Random Forest Evaluation:
              precision    recall  f1-score   support

           0       0.96      0.77      0.86     81077
           1       0.24      0.71      0.35      7950

    accuracy                           0.77     89027
   macro avg       0.60      0.74      0.61     89027
weighted avg       0.90      0.77      0.81     89027

Accuracy: 0.7676435238747795
ROC AUC: 0.7434347502409193

Since we are working with a classification problem (predicting heart failure), ROC AUC is the correct metric to evaluate our model’s performance.

## Future considerations
Logistic Regression Evaluation had nearly identical ROC AUC scores, and threshold tuning our logistic regression model at various thresholds may prove to be beneficial. Since we ran out of time, we'll leave that exercises to the open source community or our future, curious selves.

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
