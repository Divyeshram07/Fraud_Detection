import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

np.random.seed(42)

n_samples = 10000

amount = np.random.normal(5000, 2000, n_samples)
transaction_time = np.random.uniform(0, 24, n_samples)
account_age = np.random.normal(5, 2, n_samples)
device_type = np.random.choice([0, 1], n_samples)  

data = pd.DataFrame({
    "Amount": amount,
    "Transaction_Hour": transaction_time,
    "Account_Age": account_age,
    "Device_Type": device_type
})

data["Fraud"] = 0

n_fraud = 300

fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)

data.loc[fraud_indices, "Amount"] = np.random.normal(20000, 5000, n_fraud)
data.loc[fraud_indices, "Transaction_Hour"] = np.random.uniform(0, 5, n_fraud)
data.loc[fraud_indices, "Account_Age"] = np.random.normal(0.5, 0.3, n_fraud)
data.loc[fraud_indices, "Fraud"] = 1

print("Dataset Shape:", data.shape)
print(data.head())

features = ["Amount", "Transaction_Hour", "Account_Age", "Device_Type"]
X = data[features]
y = data["Fraud"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=100,
    contamination=0.03,
    random_state=42
)

model.fit(X_scaled)

pred = model.predict(X_scaled)

pred = np.where(pred == -1, 1, 0)

print("\nConfusion Matrix:")
print(confusion_matrix(y, pred))

print("\nClassification Report:")
print(classification_report(y, pred))

plt.figure(figsize=(8,6))
sns.scatterplot(x=data["Amount"], y=data["Transaction_Hour"],
                hue=pred, palette=["blue", "red"])
plt.title("Fraud Detection Visualization")
plt.xlabel("Transaction Amount")
plt.ylabel("Transaction Hour")
plt.show()