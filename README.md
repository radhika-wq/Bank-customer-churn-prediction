# 🏦 Bank Customer Churn Prediction Dashboard

An end-to-end Machine Learning web application that predicts whether a bank customer is likely to leave the bank using customer behavioral and financial data.

Built using:
- Machine Learning
- Streamlit
- Scikit-learn
- Plotly

---

# 📌 Project Overview

Customer churn is a major challenge for banks and financial institutions.  
This project helps identify customers who are at high risk of leaving by analyzing:

- Transaction behavior
- Credit usage
- Customer engagement
- Relationship history
- Financial activity

The application provides:
- Real-time churn prediction
- Churn probability score
- Interactive analytics dashboard
- Visual customer insights

---

# 🚀 Features

## ✅ Machine Learning Prediction
Predicts whether a customer is likely to churn using a trained Random Forest model.

## 📊 Interactive Dashboard
Displays:
- Total customers
- Existing vs churned customers
- Churn distribution charts

## 🧠 Intelligent Data Processing
Uses:
- Feature scaling
- One-hot encoding
- Scikit-learn pipelines
- ColumnTransformer preprocessing

## ⚡ Real-Time Prediction
Users can enter customer details and instantly receive:
- Churn probability
- Risk classification

---

# 🧠 Machine Learning Workflow

## 1. Data Preprocessing
- Cleaned and prepared dataset
- Encoded categorical features
- Scaled numerical features

## 2. Train-Test Split
Dataset divided into:
- Training set
- Testing set

## 3. Model Training
Used:
- Random Forest Classifier

## 4. Model Evaluation
Evaluated using:
- ROC-AUC Score
- Classification Report
- Confusion Matrix

---

# 📊 Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Existing Customer (0) | 0.97 | 0.97 | 0.97 |
| Attrited Customer (1) | 0.85 | 0.83 | 0.84 |

Overall Accuracy: **95%**

ROC-AUC Score: **0.98**

---

# 📈 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC Score | 0.98 |

The model achieved strong predictive performance on unseen customer data.

---

# 🛠 Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| ML Framework | Scikit-learn |
| Dashboard | Streamlit |
| Data Analysis | Pandas |
| Visualization | Plotly |
| Model Storage | Joblib |

---

# 📂 Project Structure

```bash
Bank Customer Churn Prediction/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── bank_churn.csv
│
├── models/
│   └── churn_model.pkl
│
└── src/
    └── train_model.py
```

---

# ▶️ Installation & Run

## Clone Repository

```bash
git clone https://github.com/radhika-wq/bank-customer-churn-prediction.git
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

---

# 📊 Dashboard Preview

Features included in dashboard:
- Customer analytics
- Churn prediction interface
- Probability scoring
- Interactive visualizations

---

# 🔮 Future Improvements

- Batch CSV prediction
- Cloud deployment
- Customer retention recommendations
- Explainable AI insights
- Multiple model comparison
- Authentication system

---

# 🎥 Demo Video

[🎥 Watch Project Demo](https://drive.google.com/file/d/1vW3E6Ns0lyJNZj_1nQBCUG6RLRCYELhD/view?usp=sharing)

# 👩‍💻 Author

## Radhika (24117108)


GitHub:
https://github.com/radhika-wq
