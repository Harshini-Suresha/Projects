# Diagnosing and Managing Various Diseases Using Top 5 Prescribed Drugs  
### A Patient-Centric Data-Driven Approach  

---

## Table of Contents

| Sl. # | Topic | Page # |
|:------:|:------|:------:|
| 1 | [Introduction](#-introduction) | 2 |
| 2 | [Detailed Column Explanation](#-detailed-column-explanation) | 2 |
| 3 | [Application in a Drug and Disease Diagnosis Workflow Pipeline](#-application-in-a-drug-and-disease-diagnosis-workflow-pipeline) | 4 |
| 4 | [Proposed Model](#-proposed-model) | 4 |
| 5 | [Libraries Used in Jupyter Notebook](#-libraries-used-in-jupyter-notebook) | 6 |
| 6 | [Data Cleaning and Visualization](#-data-cleaning-and-visualization) | 8 |
| 7 | [Training and Testing the Model](#-training-and-testing-the-model) | 13 |
| 8 | [Streamlit Web Application Overview](#-streamlit-web-application-overview) | 16 |

---

## Introduction
This project integrates **real-world patient feedback**, **drug usage information**, **disease conditions**, and **treatment outcomes** to create a robust data-driven healthcare prediction system.  

The objective is to:
- Diagnose diseases using patient-level attributes.  
- Identify the **top 5 most prescribed or effective drugs** for each disease.  
- Leverage **machine learning and analytics** to enhance clinical decision-making.  
- Support **personalized treatment plans** using patient response data.  

This study adopts a **patient-centric approach**, aligning computational predictions with real-world therapeutic outcomes and improving healthcare quality through data-driven insights.

**🔗 Dataset Link:** [Access the dataset here](https://drive.google.com/file/d/19t2qPNKnJAD1w17z_Fo6s6kvB5o01Ue1/view?usp=sharing)

---

## 🧾 Detailed Column Explanation
Each column in the dataset represents a unique parameter that helps establish the relationship between disease diagnosis, treatment, and drug efficacy.

| Column Name | Description |
|--------------|-------------|
| `Patient_ID` | Unique identifier for each patient record |
| `Age` | Age of the patient |
| `Gender` | Gender (Male/Female/Other) |
| `Disease` | Primary diagnosed condition |
| `Drug_Name` | Name of the prescribed drug |
| `Dosage` | Dosage prescribed for the patient |
| `Duration` | Duration of treatment (days/weeks) |
| `Side_Effects` | Reported side effects by the patient |
| `Effectiveness_Score` | Numeric score (1–10) representing treatment effectiveness |
| `Satisfaction_Score` | Overall patient satisfaction rating |
| `Review_Text` | Textual feedback or comment from the patient |
| `Sentiment` | Derived sentiment from the review (Positive/Neutral/Negative) |

---

## Application in a Drug and Disease Diagnosis Workflow Pipeline
This dataset forms the basis for a **predictive healthcare pipeline**, structured as follows:

1. **Data Collection & Integration** – Combine patient feedback, prescriptions, and treatment outcomes.  
2. **Data Preprocessing** – Clean, encode, and normalize all categorical and numerical data.  
3. **Exploratory Data Analysis (EDA)** – Visualize relationships between diseases, drugs, and patient satisfaction.  
4. **Machine Learning Model** – Predict the most effective drug for a given disease.  
5. **Evaluation & Optimization** – Use metrics like Accuracy, Precision, Recall, and F1-score.  
6. **Web Application Deployment** – Develop a Streamlit interface for end-user access.

---

## Proposed Model
The proposed system follows a **multi-step predictive modeling approach**:

1. **Input:** Disease name, patient demographics, and symptoms.  
2. **Feature Processing:** Encode and transform the input for model prediction.  
3. **Prediction:** The ML model suggests the **Top 5 most prescribed or effective drugs** for that disease.  
4. **Output:** Displays drug names, recommended dosage, and patient satisfaction probability.  

### Machine Learning Models Used:
- Logistic Regression  
- Random Forest Classifier  
- Decision Tree  
- Naïve Bayes  
- XGBoost  

The **Random Forest model** yielded the most reliable accuracy and interpretability.

---

## Libraries Used in Jupyter Notebook
| Category | Libraries |
|-----------|------------|
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Machine Learning | `scikit-learn`, `xgboost`, `lightgbm` |
| NLP Processing | `nltk`, `textblob`, `wordcloud` |
| Model Persistence | `joblib`, `pickle` |
| Web App | `streamlit` |

---

## Data Cleaning and Visualization
### Steps:
1. **Handling Missing Values** – Replace or drop null entries.  
2. **Encoding Categorical Data** – Convert gender, disease, and drug names to numeric form.  
3. **Sentiment Analysis** – Use NLP to analyze patient reviews and derive polarity scores.  
4. **Correlation Analysis** – Identify key variables influencing drug effectiveness.  
5. **Visualization Examples:**
   - Distribution of drugs per disease  
   - Average satisfaction score by drug  
   - Word clouds of positive and negative reviews  
   - Heatmaps showing feature correlations  

---

## Training and Testing the Model
1. Split data into **training (80%)** and **testing (20%)** sets.  
2. Apply model training using:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
3. Evaluate with classification metrics:
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

Use feature importance to interpret the most influential predictors.

4. Save the trained model:
```python
import joblib
joblib.dump(model, "drug_disease_model.pkl")
```
---

### Streamlit Web Application Overview

The Streamlit application provides a user-friendly interface for real-time drug prediction and disease diagnosis.

Features:

Input disease or symptoms directly.

Predict Top 5 recommended drugs with corresponding effectiveness scores.

Display visual analytics (bar charts, drug comparisons, satisfaction levels).

Option to upload new patient data for batch predictions.

Run the app:
```bash
streamlit run app.py
```
