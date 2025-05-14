import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- Load and Preprocess Data --------------------
df = pd.read_csv("Customer-Churn.csv")
df.replace(" ", np.nan, inplace=True)

# Convert numeric columns
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=['TotalCharges', 'Churn'], inplace=True)
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

if 'SeniorCitizen' in df.columns:
    df.rename(columns={"SeniorCitizen": "senior_citizen"}, inplace=True)
    df['senior_citizen'] = df['senior_citizen'].astype(str)

# -------------------- Encode Categorical Features --------------------
categorical_columns = [
    'gender', 'senior_citizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

label_encoders = {}
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

joblib.dump(label_encoders, "label_encoders.pkl")

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# -------------------- Split Data --------------------
X = df.drop(columns=['Churn'])
y = df['Churn']

# Impute missing numeric values
num_cols = X.select_dtypes(include=['number']).columns
imputer = SimpleImputer(strategy="median")
X[num_cols] = imputer.fit_transform(X[num_cols])
X = X.dropna(axis=1, how='all')

# Save column names for future inference
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

# Final train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Final imputation again in case of missed values
imputer_final = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer_final.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer_final.transform(X_test), columns=X_test.columns)

# -------------------- SMOTE + Scaling --------------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# -------------------- Train Model --------------------
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "churn_model.pkl")

# -------------------- Evaluate --------------------
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="coolwarm")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("✅ Model, scaler, encoders, and feature names saved successfully.")
