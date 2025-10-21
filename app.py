import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from catboost import CatBoostClassifier

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

required_columns = [
    'transaction_date', 'type', 'amount',
    'old_balance', 'new_balance', 'branch',
    'currency', 'device', 'location'
]

def prepare_data(df):
    print("Original columns:", df.columns.tolist())

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    df["hour"] = pd.to_datetime(df["transaction_date"]).dt.hour
    df["dayofweek"] = pd.to_datetime(df["transaction_date"]).dt.dayofweek
    df = df.drop(columns=["transaction_date"], errors="ignore")

    df["balance_diff"] = df["old_balance"] - df["new_balance"]
    df["amount_ratio"] = df["amount"] / (df["old_balance"] + 1e-6)

    final_columns = [
        'type', 'amount', 'old_balance', 'new_balance',
        'branch', 'currency', 'device', 'location',
        'hour', 'dayofweek', 'balance_diff', 'amount_ratio'
    ]

    print("✅ Final columns for model:", final_columns)
    return df[final_columns]


@app.route("/predict", methods=["POST"])
def predict_route():
    if 'file' not in request.files:
        return jsonify({"error": "Data file required"}), 400

    data_file = request.files['file']
    if data_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not (data_file.filename.endswith('.csv') or data_file.filename.endswith('.xlsx')):
        return jsonify({"error": "Only CSV and Excel files are supported"}), 400

    data_path = os.path.join(UPLOAD_FOLDER, data_file.filename)
    data_file.save(data_path)

    try:
        print(f"📂 Processing file: {data_path}")
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        else:
            df = pd.read_excel(data_path)

        print(f"✅ File loaded successfully with {len(df)} rows")

        df_prepared = prepare_data(df.copy())

        model = CatBoostClassifier()
        model.load_model("catboost_model.cbm")
        print("🤖 Model loaded successfully.")

        # ✅ توحيد الأعمدة حسب ما تدرب عليه المودل
        model_features = model.feature_names_
        print("🧩 Model was trained on features:", model_features)

        # أضف الأعمدة الناقصة بقيم صفر أو None
        for col in model_features:
            if col not in df_prepared.columns:
                df_prepared[col] = 0

        # تأكد من ترتيب الأعمدة بنفس ترتيب التدريب
        df_prepared = df_prepared[model_features]
        print("✅ Final aligned columns:", df_prepared.columns.tolist())

        preds = model.predict(df_prepared)
        probas = model.predict_proba(df_prepared)[:, 1]

        df["predicted_fraud"] = preds
        df["fraud_probability"] = np.round(probas * 100, 2)

        result_data = df.head(50).replace({np.nan: None}).to_dict(orient="records")

        response = {
            "success": True,
            "total_count": len(df),
            "fraud_count": int((df["predicted_fraud"] == 1).sum()),
            "fraud_rate": round((df["predicted_fraud"] == 1).mean() * 100, 2),
            "data": result_data
        }

        print(f"✅ Prediction complete: {response['fraud_count']} fraud cases out of {response['total_count']}")
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("🔥 Error details:", str(e))
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(data_path):
            os.remove(data_path)


@app.route("/")
def serve_home():
    return send_from_directory('.', 'home.html')

@app.route("/dashboard")
def serve_dashboard():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


if __name__ == "__main__":
    print("🚀 Starting Fraud Detection Server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
