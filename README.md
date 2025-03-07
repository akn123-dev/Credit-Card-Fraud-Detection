# Credit-Card-Fraud-Detection
Credit Card Fraud Detection

## Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset consists of anonymized transaction data, where the goal is to classify transactions as either fraudulent or legitimate.

## Dataset

The dataset contains features representing credit card transactions.

Due to confidentiality, feature names are anonymized.

The target variable (Class) indicates whether a transaction is fraudulent (1) or legitimate (0).

# Methodology

## Data Preprocessing:

Handling missing values (if any)

Data normalization/scaling

Balancing the dataset (e.g., using oversampling or undersampling)

Exploratory Data Analysis (EDA):

Visualizing data distribution

Identifying trends and anomalies

## Model Training:

Supervised learning models such as:

Logistic Regression

Random Forest

Support Vector Machines (SVM)

Performance evaluation using metrics like:

Accuracy

Precision, Recall, F1-score

AUC-ROC Curve

## Model Interpretation:


## Dependencies

Ensure you have the following Python libraries installed:

pip install pandas numpy scikit-learn matplotlib seaborn

How to Run the Notebook

Open the Jupyter Notebook environment.

Load the CC_Fraud_detection.ipynb file.

Execute the cells sequentially to preprocess data, train models, and analyze results.

## Results

The trained model achieves a high recall score to minimize false negatives.

Feature importance analysis helps in understanding transaction patterns leading to fraud detection.

## Future Improvements

Implement deep learning models for better accuracy.

Deploy the model using Flask or FastAPI for real-time fraud detection.

Optimize the model using hyperparameter tuning.

Author

Akhil Nair
