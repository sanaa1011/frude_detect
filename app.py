import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
from catboost import CatBoostClassifier

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# الفيتشرز الجديدة (نفس التي دربت عليها المودل)
expected_features = {
    "user_id": "user_id",
    "transaction_date": "transaction_date", 
    "type": "type",
    "amount": "amount",
    "old_balance": "old_balance",
    "new_balance": "new_balance",
    "destination_account": "destination_account",
    "source_account": "source_account",
    "branch": "branch",
    "currency": "currency",
    "device": "device", 
    "ip": "ip",
    "location": "location"
}

# الأعمدة التصنيفية
cat_features = ["user_id", "type", "destination_account", "source_account", 
                "branch", "currency", "device", "ip", "location"]

def prepare_data(df, expected_features):
    """
    تحضير البيانات لتتناسب مع المودل المدرب
    """
    print("Original columns:", df.columns.tolist())
    
    # إعادة تسمية الأعمدة إذا كانت مختلفة
    df = df.rename(columns=expected_features)
    print("After renaming columns:", df.columns.tolist())
    
    # استخراج الميزات من التاريخ
    if "transaction_date" in df.columns:
        df["hour"] = pd.to_datetime(df["transaction_date"]).dt.hour
        df["day_of_week"] = pd.to_datetime(df["transaction_date"]).dt.dayofweek
        df = df.drop(columns=["transaction_date"])
    else:
        # إذا لم يكن هناك تاريخ، استخدم قيم افتراضية
        df["hour"] = 12
        df["day_of_week"] = 1
    
    # ترتيب الأعمدة كما في التدريب
    final_columns = [
        "user_id", "type", "amount", "old_balance", "new_balance",
        "destination_account", "source_account", "branch", "currency", 
        "device", "ip", "location", "hour", "day_of_week"
    ]
    
    # التأكد من وجود جميع الأعمدة
    for col in final_columns:
        if col not in df.columns:
            # إضافة الأعمدة المفقودة بقيم افتراضية
            if col in ["amount", "old_balance", "new_balance", "hour", "day_of_week"]:
                df[col] = 0.0
            else:
                df[col] = "unknown"
    
    print("Final columns for model:", df[final_columns].columns.tolist())
    return df[final_columns]

@app.route("/predict", methods=["POST"])
def predict_route():
    if 'file' not in request.files:
        return jsonify({"error": "Data file required"}), 400

    data_file = request.files['file']
    if data_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # التحقق من نوع الملف
    if not (data_file.filename.endswith('.csv') or data_file.filename.endswith('.xlsx')):
        return jsonify({"error": "Only CSV and Excel files are supported"}), 400

    data_path = os.path.join(UPLOAD_FOLDER, data_file.filename)
    data_file.save(data_path)

    try:
        # قراءة الملف
        print(f"Processing file: {data_path}")
        
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        else:
            df = pd.read_excel(data_path)

        print(f"File loaded successfully with {len(df)} rows")
        print(f"Columns in file: {df.columns.tolist()}")
        
        # الفيتشرز المطلوبة (نفس التي دربت عليها المودل)
        required_columns = [
            'user_id', 'transaction_date', 'type', 'amount', 
            'old_balance', 'new_balance', 'destination_account', 
            'source_account', 'branch', 'currency', 'device', 'ip', 'location'
        ]
        
        # التحقق من الأعمدة المفقودة
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            available_cols = df.columns.tolist()
            error_msg = f"Missing columns: {', '.join(missing_cols)}. Available columns: {', '.join(available_cols)}"
            print(f"Error: {error_msg}")
            return jsonify({"error": error_msg}), 400

        print("All required columns are present")

        # تحضير البيانات للمودل
        print("Preparing data for model...")
        df_prepared = prepare_data(df.copy(), expected_features)
        print("Data preparation completed")
        
        # محاكاة المودل (يمكنك استبدال هذا بالمودل الحقيقي)
        print("Making predictions...")
        np.random.seed(42)
        preds = np.random.randint(0, 2, len(df))
        probas = np.random.random(len(df))

        # إضافة النتائج للبيانات الأصلية
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
        
        print(f"Prediction complete: {response['fraud_count']} fraud cases out of {response['total_count']} transactions")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
    finally:
        if os.path.exists(data_path):
            os.remove(data_path)

# Routes for serving different pages
@app.route("/")
def serve_home():
    print("Serving home page...")
    try:
        return send_from_directory('.', 'home.html')
    except Exception as e:
        print(f"Error serving home page: {e}")
        return "Home page not found", 404

@app.route("/dashboard")
def serve_dashboard():
    print("Serving dashboard page...")
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        print(f"Error serving dashboard: {e}")
        return "Dashboard not found", 404

@app.route('/<path:path>')
def serve_static(path):
    print(f"Serving static file: {path}")
    return send_from_directory('.', path)

if __name__ == "__main__":
    print("Starting FraudGuard Server...")
    print("Home Page: http://localhost:5000/")
    print("Dashboard: http://localhost:5000/dashboard")
    print("Current directory:", os.getcwd())
    print("Files in directory:", [f for f in os.listdir('.') if f.endswith('.html')])
    app.run(debug=True, host='0.0.0.0', port=5000)