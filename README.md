# Fetal Health Classification & Clinical Diagnostics Pipeline

## 1. Project Overview
This project leverages machine learning to classify fetal health status into three clinical categories using cardiotocogram (CTG) metrics:
1. **Normal (1.0)**: Healthy, reassuring fetal state.
2. **Suspect (2.0)**: Warning signs present; requires close monitoring.
3. **Pathological (3.0)**: High risk of fetal distress; requires immediate clinical action.

Our objective is to train, tune, and evaluate several classifiers (Logistic Regression, Decision Trees, Random Forests, and XGBoost) while optimizing for clinical safety by minimizing **False Reassurances** (misclassifying at-risk fetuses as normal).

---

## 2. Model Performance Summary

Based on out-of-sample evaluation on our test dataset ($426$ cases: $332$ Normal, $59$ Suspect, $35$ Pathological), the trained and optimized models performed as follows:

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **90.61%** | 81.89% | **81.81%** | 81.80% | **0.9660** |
| **Random Forest** | 90.38% | **84.33%** | 81.25% | **82.51%** | 0.9516 |
| **Decision Tree** | 86.38% | 76.54% | 71.80% | 73.35% | 0.9301 |
| **Logistic Regression** | 85.92% | 81.48% | 60.60% | 66.90% | 0.9354 |

* **XGBoost** achieves the highest overall accuracy ($90.61\%$) and discrimination power with an outstanding Area Under the ROC Curve of **$0.9660$**.
* **Random Forest** achieves the highest macro F1-Score ($82.51\%$) and overall macro precision.

---

## 3. Clinical Error Analysis
In fetal monitoring, predictions are not just mathematical scores—they represent lives. We analyze two critical types of diagnostic errors:

### **A. False Reassurance (The High-Risk Miss) ⚠️**
* **Definition**: Predicting a fetus is **Normal** when it is actually **Suspect** or **Pathological**.
* **XGBoost**: **15 missed cases** (Lowest Risk)
* **Random Forest**: 18 missed cases
* **Decision Tree**: 27 missed cases
* **Logistic Regression**: **50 missed cases** (Highest Risk)

### **B. False Alarm (The Over-Diagnosis) 🔔**
* **Definition**: Predicting a fetus is **Suspect** or **Pathological** when it is actually **Normal**.
* **XGBoost**: 17 alarms
* **Random Forest**: 17 alarms
* **Decision Tree**: 24 alarms
* **Logistic Regression**: **4 alarms**

> **Clinical Verdict**: While Logistic Regression has almost no false alarms, it is clinically dangerous because it falsely reassures parents in 50 cases. **XGBoost** is the safest model for clinical triage, successfully detecting over **84%** of abnormal cases with minimal dangerous misses.

---

## 4. How to Run the Pipeline & API

### **Prerequisites**
Make sure you have the required dependencies installed:
```bash
pip install pandas numpy scikit-learn xgboost joblib fastapi uvicorn pydantic
```

### **A. Training & Serializing the Model**
Run the Jupyter notebook cells sequentially to clean the data, execute Stratified 5-Fold Grid Searches, and save the final pipeline:
```python
# Serializes the entire pipeline (including vectorizer, scaler, and model)
joblib.dump({"model": final_xgb}, "../models/xgboost_pipeline.joblib")
```

### **B. Starting the FastAPI Production Server**
A FastAPI application is configured to expose this pipeline. To start the API server, navigate to your app directory and execute:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### **C. Making a Sample Prediction Request**
Send a POST request to `/predict` with the 7 critical cardiotocography features:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "prolongued_decelerations": 0.002,
  "accelerations": 0.000,
  "uterine_contractions": 0.007,
  "light_decelerations": 0.006,
  "fetal_movement": 0.001,
  "severe_decelerations": 0.000,
  "baseline_value": 129.0
}'
```
