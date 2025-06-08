# 🩺 Diabetes Prediction using Machine Learning

This project presents a machine learning-based predictive model to forecast the risk of diabetes using healthcare data. It includes end-to-end implementation — from data preprocessing and EDA to model training and time-series forecasting. The study compares three regression-based algorithms and applies techniques like SMOTE, feature importance analysis, and model performance evaluation.

---

## 📌 Project Objectives

- Predict diabetes likelihood using clinical features.
- Compare the effectiveness of multiple machine learning models.
- Address class imbalance using SMOTE.
- Visualize trends and patterns in healthcare data.
- Perform forecasting on billing data using ARIMA and Exponential Smoothing.

---

## 📊 Dataset Overview

| Feature                | Description                                   |
|------------------------|-----------------------------------------------|
| Pregnancies            | Number of times pregnant                      |
| Glucose                | Plasma glucose concentration                  |
| BloodPressure          | Diastolic blood pressure                      |
| SkinThickness          | Triceps skin fold thickness                   |
| Insulin                | Serum insulin level                           |
| BMI                    | Body mass index                               |
| DiabetesPedigreeFunction | Genetic predisposition factor              |
| Age                    | Age of the patient                            |
| Outcome                | 1 = diabetic, 0 = non-diabetic                |

**Source**: UCI Machine Learning Repository

---

## 🧪 Tools & Technologies

- **Language**: Python
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `SMOTE`, `statsmodels`
- **Models Used**:
  - Random Forest Regressor
  - Logistic Regression
  - Bagging Regressor
- **Forecasting Models**:
  - ARIMA
  - Exponential Smoothing

---

## 🔍 Exploratory Data Analysis (EDA)

- Univariate and bivariate visualizations
- Correlation heatmap and distribution plots
- Boxplots for outlier detection
- Histograms for feature distributions

---

## ⚙️ Data Preprocessing

- Missing value imputation
- Feature scaling and normalization
- Categorical encoding
- Synthetic Minority Over-sampling Technique (SMOTE) for class balancing
- Train-test split (80-20)

---

## 🤖 Model Implementation & Evaluation

| Model                 | Accuracy | MAE     | MSE     | RMSE    |
|----------------------|----------|---------|---------|---------|
| Random Forest         | 95%      | 0.0008  | 0.025   | 0.0378  |
| Bagging Regressor     | 88%      | ~0.005  | ~0.06   | ~0.08   |
| Logistic Regression   | 85%      | ~0.006  | ~0.08   | ~0.09   |

**Evaluation Metrics**:
- Accuracy
- Precision / Recall / F1-Score
- Confusion Matrix
- MAE, MSE, RMSE
- Cross-validation (5-fold)

---

## 📈 Time Series Forecasting

Used **billing data trends** to forecast future expenditure:

- 📉 **Exponential Smoothing**: for smooth trend line forecast.
- 🔁 **ARIMA**: for seasonality and oscillation modeling.

---

## 🔑 Key Insights

- **Random Forest** performed best due to its ensemble nature and ability to model non-linear patterns.
- **SMOTE** significantly improved model fairness and accuracy on minority class.
- **Forecasting** provides useful insight for health management and planning.

---

## 🧠 Future Improvements

- Integrate SHAP or LIME for explainability
- Add additional features like lifestyle and demographic info
- Deploy model as a web app (Flask or Streamlit)
- Try deep learning models (e.g., ANN, RNN)
- Perform real-time monitoring on health data

---

## 📂 Project Structure

```bash
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── diabetes_prediction.ipynb
├── models/
│   └── saved_model.pkl
├── images/
│   └── confusion_matrix.png
├── README.md
├── requirements.txt
└── LICENSE
