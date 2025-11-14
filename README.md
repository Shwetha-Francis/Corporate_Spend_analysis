AI-Powered Spend Intelligence Using ML & Deep Learning

This project builds a complete AI-powered Spend Intelligence system that performs:

✔️ Data Cleaning & Preprocessing

✔️ NLP Feature Engineering (TF-IDF + Word2Vec)

✔️ Expense Category Classification

✔️ Time-Series Forecasting

✔️ Anomaly Detection

✔️ ML vs DL Performance Comparison

The goal is to automate spend analytics using modern machine learning and deep learning methods.

📂 Project Workflow
1. Data Cleaning

Removed duplicates

Parsed dates

Converted numeric fields (Quantity, UnitPrice, TotalCost)

Recomputed TotalCost when missing

Handled mismatches using tolerance

Filled missing values

Removed invalid rows

2. Feature Engineering

Date features (Year, Month, Quarter, Weekday)

SpendPerUnit

SupplierFrequency

BuyerFrequency

Rolling Spend (7-day, 30-day)

Category-level spend aggregation

IQR-based outlier flag

3. NLP Feature Extraction
TF-IDF

Max features: 300

n-grams: 1–2

Creates sparse text vectors

Word2Vec

vector_size = 100

Trained on item names

Sentence-level embedding = average token vectors

4. Classification Models

The task: Predict expense Category using numeric + TF-IDF + Word2Vec features.

Models used:

Logistic Regression

Random Forest

XGBoost

Deep Learning MLP

Metrics:

Accuracy

Macro F1-score

Results are plotted as comparison bar charts.

5. Forecasting Models

Goal: Predict future monthly spend per category.

Models used:

RandomForestRegressor

LSTM Model

Evaluated using RMSE.

6. Anomaly Detection

Three techniques:

Isolation Forest

One-Class SVM

Autoencoder (DL)

Autoencoder reconstruction error threshold (99th percentile) used as anomaly score.

📊 Outputs

Classification accuracy comparison (bar chart)

Forecasting RMSE comparison (bar chart)

Autoencoder MSE distribution (histogram)

Anomaly flags added to dataset
