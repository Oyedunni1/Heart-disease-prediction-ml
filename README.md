# ❤️ Heart Disease Prediction using Machine Learning

This project explores the use of machine learning models to predict the presence of heart disease based on patient medical data. The goal is to enable early detection and support clinical decision-making through data-driven insights.

---

## 🧠 Project Objective

To train and compare the performance of five machine learning models in predicting heart disease, and deploy the models using lightweight APIs for real-time inference.

---

## 📊 Dataset

- **Source**: [Kaggle Heart Disease Dataset](https://fedesoriano/heart-failure-prediction)
- **Features**: 11 clinical attributes including age, cholesterol, blood pressure, etc.
- **Target**: Presence (1) or absence (0) of heart disease

---

## 🔍 Models Used

| Model            | Accuracy (example) |
|------------------|--------------------|
| XGBoost          | 88.5%              |
| K-Nearest Neighbors (KNN) | 84.2%      |
| Support Vector Classifier (SVC) | 86.7% |
| Gaussian Naive Bayes | 82.1%         |
| Perceptron       | 79.6%              |

> The models were trained, evaluated, and cross-validated to ensure stability and reliability.

---

## 🚀 Deployment

Two lightweight deployment options were built:

### 🔸 Flask API

- Accepts patient input data in JSON format
- Returns a prediction (0 or 1)
- Easy to integrate into healthcare dashboards

```bash
POST /predict
{
 "age": 54,
  "sex": 1,
  "cp": 0,
  "trestbps": 130,
  "chol": 246,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.5,
  "slope": 2,
  "ca": 0,
  "thal": 2
}


