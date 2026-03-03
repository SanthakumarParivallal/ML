# 💳 Credit Card Fraud Detection Using Machine Learning

### Reimplementation & Comparative Analysis

------------------------------------------------------------------------

## 📌 Project Overview

This project is a complete reimplementation of the research paper:

**"Credit Card Fraud Detection Using Machine Learning"**\
Sailusha et al., 2020 (ICICCS 2020, IEEE)

The goal is to detect fraudulent credit card transactions using Machine
Learning algorithms and compare their performance.

We strictly used the same dataset mentioned in the paper and reproduced
the methodology for fair comparison.

------------------------------------------------------------------------

## 📊 Dataset Information

-   Source: Kaggle -- European Cardholders Dataset
-   Transactions: 284,807
-   Fraud Cases: 492
-   Fraud Percentage: 0.172%
-   Highly Imbalanced Dataset

### Features:

-   V1 -- V28 (PCA transformed features)
-   Time
-   Amount
-   Class (Target Variable)
    -   0 → Legitimate
    -   1 → Fraud

Dataset file used: creditcard.csv

------------------------------------------------------------------------

## 🧠 Algorithms Implemented

-   Random Forest
-   AdaBoost

Implemented using scikit-learn.

------------------------------------------------------------------------

## 📈 Evaluation Metrics

### Metrics used in Original Paper:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion Matrix
-   ROC Curve

### Additional Metrics (Our Contribution):

-   ROC-AUC
-   Precision--Recall Curve
-   Matthews Correlation Coefficient (MCC)
-   Balanced Accuracy
-   Average Precision Score

------------------------------------------------------------------------

## 📊 Results (Reimplementation)

### 🔹 Random Forest

  Accuracy- 0.9995, 
  Precision- 0.9573,
  Recall- 0.7568,
  F1 Score- 0.8453,
  ROC-AUC- 0.9307,
  MCC- 0.8509.

------------------------------------------------------------------------

### 🔹 AdaBoost

  Accuracy- 0.9990,
  Precision- 0.7353,
  Recall- 0.6757,
  F1 Score- 0.7042,
  ROC-AUC- 0.9675,
  MCC- 0.7044.

------------------------------------------------------------------------

## 💾 Saved Models

saved_models/random_forest_fraud_model.pkl
saved_models/adaboost_fraud_model.pkl

Load example:

``` python
import joblib
model = joblib.load("saved_models/random_forest_fraud_model.pkl")
prediction = model.predict(X_sample)
```

------------------------------------------------------------------------

## 🚀 How to Run

Install dependencies:

``` python
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

Run:

``` python
python credit_card_fraud_detection.py
```

Or open the notebook:

Credit_Card_Fraud_Detection.ipynb

------------------------------------------------------------------------

## 🔍 Key Findings

-   Dataset is extremely imbalanced.
-   Accuracy alone is misleading.
-   Random Forest performs better overall.
-   Ensemble methods work well for fraud detection.

------------------------------------------------------------------------

## 🔬 Proposed Improvements

-   Apply SMOTE for imbalance handling
-   Hyperparameter tuning (GridSearchCV)
-   Try advanced models (XGBoost, LightGBM)
-   Implement Deep Learning models
-   Real-time fraud detection pipeline

------------------------------------------------------------------------

## 👨‍💻 Author

### 1. Jatin Kumar
### 2. Santhakumar Parivallal
### 3. Muhammad zahid
### 4. Vineel bokkina

---

### -Machine Learning Academic Project

------------------------------------------------------------------------

### ⭐ If you found this useful, consider giving the repository a star!
