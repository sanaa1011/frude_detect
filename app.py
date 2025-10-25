import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, session, redirect
from flask_cors import CORS
from catboost import CatBoostClassifier
import mysql.connector
from mysql.connector import Error
import bcrypt
import re
import uuid
from datetime import datetime

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
app.secret_key = 'fraud_detection_secret_key_2025'

UPLOAD_FOLDER = "uploads"
USER_DATA_FOLDER = "user_data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(USER_DATA_FOLDER, exist_ok=True)

required_columns = ['user_id',
 'transaction_date',
 'type',
 'amount',
 'old_balance',
 'new_balance',
 'balance_mismatch',
 'amount_spike',
 'destination_account',
 'new_destination',
 'blacklisted_dest',
 'source_account',
 'branch',
 'currency',
 'device',
 'device_change',
 'ip',
 'ip_unusual',
 'location',
 'odd_hour',
 'velocity']
# دوال قاعدة البيانات
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='fraud_users',
            user='root',
            password=''
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except (ValueError, Exception):
        return password == hashed

# دالة لتحويل كلمات المرور القديمة إلى مشفرة
def migrate_passwords():
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT id, pass FROM users WHERE LENGTH(pass) < 60")
            users = cursor.fetchall()
            
            for user in users:
                if len(user['pass']) < 60:
                    hashed = hash_password(user['pass'])
                    cursor.execute("UPDATE users SET pass = %s WHERE id = %s", (hashed, user['id']))
                    print(f"Updated password for user {user['id']}")
            
            connection.commit()
            print("Password migration completed")
            
        except Error as e:
            print(f"Error migrating passwords: {e}")
        finally:
            cursor.close()
            connection.close()

# دوال إدارة تجارب المستخدم
def save_user_experiment(user_id, filename, result_data, save_data=False):
    """حفظ تجربة المستخدم في قاعدة البيانات"""
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            experiment_id = str(uuid.uuid4())
            
            # حفظ البيانات إذا كان المستخدم مسجل
            if save_data and user_id:
                # حفظ ملف النتائج
                result_filename = f"{experiment_id}_results.csv"
                result_path = os.path.join(USER_DATA_FOLDER, result_filename)
                
                # تحويل البيانات إلى CSV وحفظها
                df = pd.DataFrame(result_data['data'])
                df.to_csv(result_path, index=False)
                
                cursor.execute(
                    """INSERT INTO user_experiments 
                    (id, user_id, filename, result_filename, total_count, fraud_count, fraud_rate, created_at) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (experiment_id, user_id, filename, result_filename, 
                     result_data['total_count'], result_data['fraud_count'], 
                     result_data['fraud_rate'], datetime.now())
                )
            else:
                # تجربة بدون حفظ (للضيوف)
                cursor.execute(
                    """INSERT INTO user_experiments 
                    (id, user_id, filename, total_count, fraud_count, fraud_rate, created_at, is_temporary) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (experiment_id, None, filename, 
                     result_data['total_count'], result_data['fraud_count'], 
                     result_data['fraud_rate'], datetime.now(), True)
                )
            
            connection.commit()
            return experiment_id
            
        except Error as e:
            print(f"Error saving experiment: {e}")
            return None
        finally:
            cursor.close()
            connection.close()
    return None

def get_user_experiments(user_id):
    """جلب تجارب المستخدم"""
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM user_experiments WHERE user_id = %s AND is_temporary = FALSE ORDER BY created_at DESC",
                (user_id,)
            )
            experiments = cursor.fetchall()
            return experiments
        except Error as e:
            print(f"Error getting experiments: {e}")
            return []
        finally:
            cursor.close()
            connection.close()
    return []

# دوال تحضير البيانات
def prepare_data(df):
    print("Original columns:", df.columns.tolist())

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    # df["hour"] = pd.to_datetime(df["transaction_date"]).dt.hour
    # df["dayofweek"] = pd.to_datetime(df["transaction_date"]).dt.dayofweek
    # df = df.drop(columns=["transaction_date"], errors="ignore")

    # df["balance_diff"] = df["old_balance"] - df["new_balance"]
    # df["amount_ratio"] = df["amount"] / (df["old_balance"] + 1e-6)

    final_columns = ['user_id',
 'transaction_date',
 'type',
 'amount',
 'old_balance',
 'new_balance',
 'balance_mismatch',
 'amount_spike',
 'destination_account',
 'new_destination',
 'blacklisted_dest',
 'source_account',
 'branch',
 'currency',
 'device',
 'device_change',
 'ip',
 'ip_unusual',
 'location',
 'odd_hour',
 'velocity']

    print("✅ Final columns for model:", final_columns)
    return df[final_columns]

# Routes الأساسية
@app.route("/")
def serve_home():
    return send_from_directory('.', 'home.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
                user = cursor.fetchone()
                
                if user and check_password(password, user['pass']):
                    session['user_id'] = user['id']
                    session['first_name'] = user['first_name']
                    session['email'] = user['email']
                    return jsonify({"success": True, "message": "تم تسجيل الدخول بنجاح"})
                else:
                    return jsonify({"error": "البريد الإلكتروني أو كلمة المرور غير صحيحة"}), 401
                    
            except Error as e:
                print(f"Database error: {e}")
                return jsonify({"error": f"خطأ في قاعدة البيانات: {str(e)}"}), 500
            except Exception as e:
                print(f"Login error: {e}")
                return jsonify({"error": f"خطأ في تسجيل الدخول: {str(e)}"}), 500
            finally:
                cursor.close()
                connection.close()
        else:
            return jsonify({"error": "خطأ في الاتصال بقاعدة البيانات"}), 500
    
    return send_from_directory('.', 'login.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        password = request.form['password']
        
        if not validate_email(email):
            return jsonify({"error": "البريد الإلكتروني غير صالح"}), 400
        
        if len(password) < 6:
            return jsonify({"error": "كلمة المرور يجب أن تكون 6 أحرف على الأقل"}), 400
        
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                
                cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                if cursor.fetchone():
                    return jsonify({"error": "البريد الإلكتروني مسجل مسبقاً"}), 400
                
                hashed_password = hash_password(password)
                username = f"{first_name}_{last_name}".lower()
                
                cursor.execute(
                    "INSERT INTO users (first_name, last_name, email, pass) VALUES (%s, %s, %s, %s)",
                    (first_name, last_name, email, hashed_password)
                )
                connection.commit()
                
                return jsonify({"success": True, "message": "تم إنشاء الحساب بنجاح"})
                
            except Error as e:
                print(f"Database error: {e}")
                return jsonify({"error": f"خطأ في قاعدة البيانات: {str(e)}"}), 500
            finally:
                cursor.close()
                connection.close()
        else:
            return jsonify({"error": "خطأ في الاتصال بقاعدة البيانات"}), 500
    
    return send_from_directory('.', 'register.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect('/')

@app.route("/dashboard")
def dashboard():
    if 'user_id' not in session:
        # السماح للضيوف بالدخول ولكن بدون حفظ البيانات
        session['guest'] = True
    return send_from_directory('.', 'index.html')

@app.route("/my-experiments")
def my_experiments():
    if 'user_id' not in session:
        return redirect('/login')
    return send_from_directory('.', 'experiments.html')

@app.route("/api/experiments")
def get_experiments():
    if 'user_id' not in session:
        return jsonify({"error": "يجب تسجيل الدخول"}), 401
    
    experiments = get_user_experiments(session['user_id'])
    return jsonify({"experiments": experiments})

@app.route("/predict", methods=["POST"])
def predict_route():
    if 'user_id' not in session and 'guest' not in session:
        return jsonify({"error": "يجب تسجيل الدخول أولاً"}), 401
        
    if 'file' not in request.files:
        return jsonify({"error": "Data file required"}), 400

    data_file = request.files['file']
    if data_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not (data_file.filename.endswith('.csv') or data_file.filename.endswith('.xlsx')):
        return jsonify({"error": "Only CSV and Excel files are supported"}), 400

    # الحصول على خيار الحفظ من النموذج
    save_option = request.form.get('saveOption', 'guest')
    
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
        model.load_model("fra_catboost_model.cbm")
        print("🤖 Model loaded successfully.")

        model_features = model.feature_names_
        print("🧩 Model was trained on features:", model_features)

        for col in model_features:
            if col not in df_prepared.columns:
                df_prepared[col] = 0

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
            "data": result_data,
            "filename": data_file.filename
        }

        # حفظ التجربة إذا كان المستخدم مسجل واختار الحفظ
        if 'user_id' in session and session['user_id'] and save_option == 'save':
            save_user_experiment(
                session['user_id'], 
                data_file.filename, 
                response, 
                save_data=True
            )
            print("✅ Experiment saved to user account")
        else:
            # تجربة ضيف (لا تحفظ البيانات)
            save_user_experiment(
                None, 
                data_file.filename, 
                response, 
                save_data=False
            )
            print("✅ Guest experiment (not saved)")

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

# أضف هذا الroute في app.py
@app.route("/api/auth-status")
def auth_status():
    """التحقق من حالة تسجيل الدخول"""
    if 'user_id' in session:
        return jsonify({
            "logged_in": True,
            "user_name": session.get('first_name', 'User')
        })
    else:
        return jsonify({"logged_in": False})
    
# Routes للملفات الثابتة
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


# Middleware للتحقق من تسجيل الدخول
@app.before_request
def check_auth():
    public_routes = ['/', '/login', '/register', 'home.html', 'login.html', 'register.html', 'static']
    
    if request.path in public_routes or request.path.startswith('/static'):
        return
    
    if 'user_id' not in session and 'guest' not in session:
        return redirect('/login')

if __name__ == "__main__":
    print("🚀 Starting Fraud Detection Server...")
    
    # تحويل كلمات المرور القديمة إلى مشفرة
    print("🔄 Migrating passwords...")
    migrate_passwords()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
