from flask import Flask, render_template, jsonify, request, send_file
import numpy as np
import pandas as pd
import joblib
import random
import io
import os

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
os.makedirs("model", exist_ok=True)

transactions = []
model_metrics = {}

# =========================================================
#  TABULAR FRAUD MODEL (Transaction Model)
# =========================================================

def train_tabular_model():

    np.random.seed(42)
    n_samples = 4000
    n_fraud = 400

    normal_amount = np.random.normal(8000, 2500, n_samples - n_fraud)
    fraud_amount = np.random.normal(45000, 5000, n_fraud)

    normal_hour = np.random.uniform(6, 22, n_samples - n_fraud)
    fraud_hour = np.random.uniform(0, 3, n_fraud)

    normal_account_age = np.random.uniform(2, 6, n_samples - n_fraud)
    fraud_account_age = np.random.uniform(0, 0.5, n_fraud)

    normal_device = np.ones(n_samples - n_fraud)
    fraud_device = np.zeros(n_fraud)

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

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    rf_model = RandomForestClassifier(n_estimators=120)
    rf_model.fit(X_train, y_train)

    iso_model = IsolationForest(contamination=0.1)
    iso_model.fit(X_train)

    y_pred = rf_model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3)
    }

    joblib.dump(rf_model, "model/rf_model.pkl")
    joblib.dump(iso_model, "model/iso_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    return rf_model, iso_model, scaler, metrics


# =========================================================
#  NLP FRAUD MESSAGE MODEL
# =========================================================

def train_nlp_model():

    fraud_messages = [
        "Your account has been blocked. Click here to verify.",
        "Urgent! Transfer money immediately.",
        "You won a lottery prize claim now.",
        "Your bank account suspended verify OTP.",
        "Free gift card waiting click link.",
        "Confirm your password to avoid suspension.",
        "Limited time offer claim reward now."
    ]

    legit_messages = [
        "Meeting scheduled tomorrow at 10 AM.",
        "Your salary has been credited.",
        "Project update completed successfully.",
        "Dinner at 8 PM?",
        "Invoice attached for your review.",
        "Please review the attached document.",
        "Your order has been delivered."
    ]

    texts = fraud_messages + legit_messages
    labels = [1]*len(fraud_messages) + [0]*len(legit_messages)

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    joblib.dump(model, "model/nlp_model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")

    return model, vectorizer


# =========================================================
#  LOAD MODELS
# =========================================================

try:
    rf_model = joblib.load("model/rf_model.pkl")
    iso_model = joblib.load("model/iso_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
except:
    rf_model, iso_model, scaler, model_metrics = train_tabular_model()

try:
    nlp_model = joblib.load("model/nlp_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
except:
    nlp_model, vectorizer = train_nlp_model()


# =========================================================
#  TRANSACTION PREDICTION
# =========================================================

def generate_transaction():
    is_fraud = np.random.rand() < 0.1

    if is_fraud:
        return {
            "Amount": float(np.random.uniform(35000, 60000)),
            "Hour": float(np.random.uniform(0, 3)),
            "Account_Age": float(np.random.uniform(0, 0.5)),
            "Device": 0
        }
    else:
        return {
            "Amount": float(np.random.uniform(1000, 15000)),
            "Hour": float(np.random.uniform(6, 22)),
            "Account_Age": float(np.random.uniform(2, 6)),
            "Device": 1
        }


def predict_transaction(tx):

    df = pd.DataFrame([tx])
    X_scaled = scaler.transform(df)

    rf_prob = rf_model.predict_proba(X_scaled)[0][1]
    iso_flag = 1 if iso_model.predict(X_scaled)[0] == -1 else 0

    final_score = (rf_prob * 0.7) + (iso_flag * 0.3)

    explanation = "Transaction risk evaluated using behavioral patterns."

    return float(final_score), explanation


# =========================================================
#  ROUTES
# =========================================================

@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/data")
def data():
    tx = generate_transaction()
    risk, explanation = predict_transaction(tx)

    tx["Risk"] = risk
    tx["Explanation"] = explanation

    transactions.append(tx)
    if len(transactions) > 100:
        transactions.pop(0)

    return jsonify(transactions)


@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    data = request.get_json()
    message = data.get("message", "")

    X = vectorizer.transform([message])
    probability = nlp_model.predict_proba(X)[0][1]

    prediction = "Fraud Message" if probability > 0.5 else "Legitimate Message"

    return jsonify({
        "prediction": prediction,
        "probability": round(float(probability), 3)
    })


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


if __name__ == "__main__":
    app.run()