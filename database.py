import sqlite3
import pandas as pd
from datetime import datetime
import json

class FraudDatabase:
    def __init__(self, db_path='fraud_detection.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """تهيئة قاعدة البيانات والجداول"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول المستخدمين
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # جدول التحليلات (لحفظ الإحصائيات)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_id TEXT,
                total_transactions INTEGER,
                fraud_count INTEGER,
                fraud_rate REAL,
                highest_fraud_amount REAL,
                most_common_type TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_name TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # جدول النتائج (لحفظ البيانات المهمة فقط)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                user_id TEXT,
                transaction_type TEXT,
                amount REAL,
                old_balance REAL,
                new_balance REAL,
                source_account TEXT,
                destination_account TEXT,
                is_fraud INTEGER,
                fraud_probability REAL,
                transaction_date TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, user_data, analysis_data, results_data):
        """حفظ التحليل والنتائج في قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # إدخال أو تحديث المستخدم
            if 'username' in user_data:
                cursor.execute('''
                    INSERT OR IGNORE INTO users (username, email) 
                    VALUES (?, ?)
                ''', (user_data.get('username'), user_data.get('email')))
                
                cursor.execute('SELECT id FROM users WHERE username = ?', (user_data['username'],))
                user_id = cursor.fetchone()[0]
            else:
                user_id = None
            
            # إدخال التحليل
            cursor.execute('''
                INSERT INTO analyses 
                (user_id, session_id, total_transactions, fraud_count, fraud_rate, 
                 highest_fraud_amount, most_common_type, file_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                user_data.get('session_id', 'anonymous'),
                analysis_data['total_count'],
                analysis_data['fraud_count'],
                analysis_data['fraud_rate'],
                analysis_data.get('highest_fraud_amount', 0),
                analysis_data.get('most_common_type', 'Unknown'),
                user_data.get('file_name', 'unknown')
            ))
            
            analysis_id = cursor.lastrowid
            
            # إدخال النتائج (البيانات المهمة فقط)
            for result in results_data:
                cursor.execute('''
                    INSERT INTO results 
                    (analysis_id, user_id, transaction_type, amount, old_balance, 
                     new_balance, source_account, destination_account, is_fraud, 
                     fraud_probability, transaction_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_id,
                    result.get('user_id'),
                    result.get('type'),
                    result.get('amount'),
                    result.get('old_balance'),
                    result.get('new_balance'),
                    result.get('source_account'),
                    result.get('destination_account'),
                    result.get('predicted_fraud', 0),
                    result.get('fraud_probability', 0),
                    datetime.now()
                ))
            
            conn.commit()
            return analysis_id
            
        except Exception as e:
            conn.rollback()
            print(f"Database error: {e}")
            return None
        finally:
            conn.close()
    
    def get_user_analyses(self, username):
        """جلب تحليلات المستخدم"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT a.*, u.username 
            FROM analyses a 
            LEFT JOIN users u ON a.user_id = u.id 
            WHERE u.username = ? 
            ORDER BY a.analysis_date DESC
        ''', (username,))
        
        analyses = cursor.fetchall()
        conn.close()
        return analyses
    
    def get_analysis_results(self, analysis_id, limit=1000):
        """جلب نتائج تحليل معين"""
        conn = sqlite3.connect(self.db_path)
        
        # جلب النتائج مع تحديد الحد
        results_df = pd.read_sql('''
            SELECT user_id, transaction_type, amount, old_balance, new_balance,
                   source_account, destination_account, is_fraud, fraud_probability
            FROM results 
            WHERE analysis_id = ? 
            LIMIT ?
        ''', conn, params=(analysis_id, limit))
        
        conn.close()
        return results_df

# إنشاء instance من قاعدة البيانات
db = FraudDatabase()