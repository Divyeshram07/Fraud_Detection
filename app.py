from flask import Flask, render_template, jsonify, request, send_file
import numpy as np
import pandas as pd
import joblib
import random
import io
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

os.makedirs("model", exist_ok=True)

transactions = []

# ================= TRAIN MODEL =================
def train_model():
    np.random.seed(42)

    n_samples = 6000
    n_fraud = 500

    normal_amount = np.random.normal(5000, 2000, n_samples - n_fraud)
    fraud_amount = np.random.normal(25000, 8000, n_fraud)

    normal_hour = np.random.uniform(6, 22, n_samples - n_fraud)
    fraud_hour = np.random.uniform(0, 5, n_fraud)

    normal_account_age = np.random.normal(4, 1.5, n_samples - n_fraud)
    fraud_account_age = np.random.normal(0.5, 0.4, n_fraud)

    normal_device = np.random.choice([0, 1], n_samples - n_fraud)
    fraud_device = np.random.choice([0, 1], n_fraud)

    data = pd.DataFrame({
        "Amount": np.concatenate([normal_amount, fraud_amount]),
        "Hour": np.concatenate([normal_hour, fraud_hour]),
        "Account_Age": np.concatenate([normal_account_age, fraud_account_age]),
        "Device": np.concatenate([normal_device, fraud_device]),
        "Fraud": np.concatenate([np.zeros(n_samples - n_fraud), np.ones(n_fraud)])
    })

    X = data.drop("Fraud", axis=1)
    y = data["Fraud"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2
    )

    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    joblib.dump(model, "model/fraud_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    return model, scaler


# Load existing model or train new one
try:
    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
except:
    model, scaler = train_model()


# ================= TRANSACTION GENERATION =================
def generate_transaction():
    return {
        "Amount": float(np.random.uniform(1000, 40000)),
        "Hour": float(np.random.uniform(0, 24)),
        "Account_Age": float(np.random.uniform(0, 6)),
        "Device": random.choice([0, 1]),
        "Latitude": float(np.random.uniform(-60, 60)),
        "Longitude": float(np.random.uniform(-180, 180))
    }


def predict(tx):
    df = pd.DataFrame([tx])
    X_scaled = scaler.transform(
        df[["Amount", "Hour", "Account_Age", "Device"]]
    )
    probability = model.predict_proba(X_scaled)[0][1]
    return float(probability)


# ================= ROUTES =================

@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/data")
def data():
    tx = generate_transaction()
    risk = predict(tx)
    tx["Risk"] = risk

    transactions.append(tx)
    if len(transactions) > 150:
        transactions.pop(0)

    return jsonify(transactions)


@app.route("/retrain", methods=["POST"])
def retrain():
    global model, scaler
    model, scaler = train_model()
    return jsonify({"status": "Model Retrained Successfully"})


@app.route("/download")
def download():
    df = pd.DataFrame(transactions)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="fraud_report.csv"
    )


# ================= IMPROVED TEXT FRAUD ANALYZER =================

@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    try:
        data = request.get_json()
        message = data.get("message", "").lower().strip()

        if message == "":
            return jsonify({
                "prediction": "Empty Input",
                "fraud_probability": 0
            })

        # Weighted keyword scoring
        fraud_keywords = {
            "urgent": 2,
            "lottery": 3,
            "win": 2,
            "transfer": 2,
            "password": 3,
            "otp": 3,
            "blocked": 2,
            "verify": 2,
            "click": 2,
            "free": 2,
            "suspended": 2,
            "bank": 1,
            "account": 1,
            "immediately": 2,
            "prize": 3
        }

        score = 0
        for word, weight in fraud_keywords.items():
            if word in message:
                score += weight

        # Normalize score into probability
        probability = min(score / 8, 1.0)

        prediction = "Fraud" if probability >= 0.4 else "Not Fraud"

        return jsonify({
            "prediction": prediction,
            "fraud_probability": probability
        })

    except Exception as e:
        return jsonify({
            "prediction": "Error",
            "fraud_probability": 0,
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)