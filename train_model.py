# train_model.py

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_data(n_samples=10000, n_fraud=500):
    np.random.seed(42)

    normal_amount = np.random.normal(5000, 2000, n_samples - n_fraud)
    fraud_amount = np.random.normal(25000, 7000, n_fraud)

    normal_hour = np.random.uniform(6, 22, n_samples - n_fraud)
    fraud_hour = np.random.uniform(0, 5, n_fraud)

    normal_account_age = np.random.normal(5, 2, n_samples - n_fraud)
    fraud_account_age = np.random.normal(0.5, 0.3, n_fraud)

    normal_device = np.random.choice([0,1], n_samples - n_fraud)
    fraud_device = np.random.choice([0,1], n_fraud)

    data = pd.DataFrame({
        "Amount": np.concatenate([normal_amount, fraud_amount]),
        "Hour": np.concatenate([normal_hour, fraud_hour]),
        "Account_Age": np.concatenate([normal_account_age, fraud_account_age]),
        "Device": np.concatenate([normal_device, fraud_device]),
        "Fraud": np.concatenate([np.zeros(n_samples - n_fraud), np.ones(n_fraud)])
    })

    return data.sample(frac=1).reset_index(drop=True)

# Train
data = generate_data()

X = data.drop("Fraud", axis=1)
y = data["Fraud"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=200, class_weight="balanced")
model.fit(X_train, y_train)

joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model Trained & Saved Successfully!")