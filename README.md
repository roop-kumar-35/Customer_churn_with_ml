# Customer Churn Prediction using Machine Learning
## 📌 Problem Statement

Customer churn is a significant challenge in the telecom sector where users discontinue service. This classification project aims to predict customer churn using historical customer behavior and demographic data, enabling proactive retention strategies.

## 🧠 Abstract

This project explores churn prediction through machine learning models, focusing on Random Forest and Logistic Regression. Preprocessing and feature engineering techniques were applied to enhance prediction accuracy. Random Forest achieved an F1-score of 0.56, indicating its potential for actionable business decisions.

## 🚀 Objectives

- Predict churn with high accuracy using historical telecom data
- Understand the most influential factors driving customer churn
- Provide insights to enhance customer retention strategies

## 🛠️ System Requirements

**Hardware:**
- Minimum 4 GB RAM
- Intel i3 processor or better

**Software:**
- Python 3.x
- Jupyter Notebook

**Python Libraries:**
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## 🔁 Project Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training and Evaluation
6. Deployment using Streamlit

## 📊 Dataset Description

- **Source:** Kaggle (Telecom Customer Churn dataset)
- **Size:** ~7,000 rows, 20+ features
- **Type:** Public, structured data

## 🧹 Data Preprocessing

- Removed missing values and duplicates
- Label encoded categorical features
- Scaled numerical features using `StandardScaler`
- Eliminated irrelevant and highly collinear features

## 🔎 EDA Highlights

- Customers with short tenure and high monthly charges are more likely to churn
- Services like Online Security and Tech Support influence churn behavior

## 🏗️ Feature Engineering

- Engineered features like total services used
- Binned tenure into categories (short, medium, long)
- Improved interpretability and accuracy

## 🤖 Model Building

- Algorithms Used: Random Forest, Logistic Regression
- Train-Test Split: 80-20 (stratified)
- Best Performing Model: **Random Forest**

### 📈 Model Evaluation

- **Random Forest Accuracy:** 79.35%
- **Metrics Used:** Accuracy, Precision, Recall, F1-Score

## 🌐 Deployment

- Framework: **Streamlit**
- Hosting: **Streamlit Cloud**
- [🔗 Sample App (Localhost Placeholder)](http://localhost:8501)

## 📁 Source Code

- [GitHub Repository](https://github.com/karthi-keyan-ic/Customer_churn__using_ml.git)

## 🔮 Future Enhancements

- Integrate XGBoost and LightGBM for improved accuracy
- Develop a real-time churn monitoring dashboard
- Add feedback loop for dynamic retraining

## 👨‍💻 Team Members

| Name              | Role                             |
|-------------------|----------------------------------|
| **S. Karthikeyan** | Team Lead, Visualization, Evaluation |
| Nithin R.         | Data Cleaning, EDA, Code Optimization |
| M. Dhinesh Kumar  | Data Analysis, Model Tuning       |
| B. Roopkumar      | Documentation, Reporting          |

---

> 📅 Submitted on **May 16, 2025**  
> 🏫 M.P. Nachimuthu M. Jaganathan Engineering College  
> 🎓 Department of Computer Science and Engineering

