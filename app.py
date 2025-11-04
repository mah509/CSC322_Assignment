# -*- coding: utf-8 -*-
"""
نظام توقع مرض السكري - تطبيق Flask
Diabetes Prediction System - Flask Application
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

# إنشاء تطبيق Flask - Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'diabetes-prediction-secret-key-2024')

# تحميل النموذج المدرب والمطبع - Load trained model and scaler
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ تم تحميل النموذج والمطبع بنجاح!")
except:
    model = None
    scaler = None
    print("⚠️  النموذج غير موجود. يرجى تشغيل train_model.py أولاً")

# دالة لتحويل القيم الفئوية إلى رقمية - Function to convert categorical to numerical
def convert_categorical_to_numerical(data):
    """
    تحويل القيم المدخلة من المستخدم إلى قيم رقمية مناسبة للنموذج
    Convert user input values to numerical values suitable for the model
    """
    
    # تحويل مستوى الجلوكوز - Convert Glucose level
    glucose_mapping = {
        'low': 85,        # منخفض
        'medium': 120,    # متوسط
        'high': 160       # مرتفع جداً
    }
    glucose = glucose_mapping.get(data.get('glucose', 'medium'), 120)
    
    # تحويل ضغط الدم - Convert Blood Pressure
    bp_mapping = {
        'low': 60,      # منخفض
        'normal': 75,   # طبيعي
        'high': 90      # مرتفع
    }
    blood_pressure = bp_mapping.get(data.get('blood_pressure', 'normal'), 75)
    
    # تحويل سُمك الجلد - Convert Skin Thickness
    skin_mapping = {
        'low': 15,      # منخفض
        'medium': 25,   # متوسط
        'high': 35      # مرتفع
    }
    skin_thickness = skin_mapping.get(data.get('skin_thickness', 'medium'), 25)
    
    # تحويل الإنسولين - Convert Insulin
    insulin_mapping = {
        'low': 80,      # منخفض
        'medium': 120,  # متوسط
        'high': 180     # مرتفع
    }
    insulin = insulin_mapping.get(data.get('insulin', 'medium'), 120)
    
    # تحويل مؤشر كتلة الجسم - Convert BMI
    bmi_mapping = {
        'underweight': 18.5,  # نقص الوزن
        'normal': 22.0,       # طبيعي
        'overweight': 27.0,   # زيادة الوزن
        'obese': 35.0         # سمنة
    }
    bmi = bmi_mapping.get(data.get('bmi', 'normal'), 22.0)
    
    # تحويل العمر - Convert Age
    age_mapping = {
        'young': 25,        # شاب
        'middle': 40,       # متوسط العمر
        'old': 60           # كبير السن
    }
    age = age_mapping.get(data.get('age', 'middle'), 40)
    
    # عدد مرات الحمل - Pregnancies (0 if male)
    pregnancies = int(data.get('pregnancies', 0))
    
    # وظيفة نسب السكري (قيمة افتراضية) - Diabetes Pedigree Function (default value)
    diabetes_pedigree = 0.5
    
    # إرجاع المصفوفة بالترتيب الصحيح - Return array in correct order
    # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    #  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    return np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                      insulin, bmi, diabetes_pedigree, age]])

# الصفحة الرئيسية - Home Page
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    عرض الصفحة الرئيسية ومعالجة التنبؤات
    Display home page and handle predictions
    """
    prediction_result = None
    prediction_class = None
    user_input = {}
    
    if request.method == 'POST':
        try:
            # جمع بيانات المستخدم - Collect user data
            user_input = {
                'gender': request.form.get('gender'),
                'pregnancies': request.form.get('pregnancies', 0),
                'glucose': request.form.get('glucose'),
                'blood_pressure': request.form.get('blood_pressure'),
                'skin_thickness': request.form.get('skin_thickness'),
                'insulin': request.form.get('insulin'),
                'bmi': request.form.get('bmi'),
                'age': request.form.get('age')
            }
            
            # تحويل البيانات إلى أرقام - Convert data to numbers
            features = convert_categorical_to_numerical(user_input)
            
            # تطبيع البيانات - Scale features
            if scaler:
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features
            
            # التنبؤ - Make prediction
            if model:
                prediction = model.predict(features_scaled)[0]
                
                if prediction == 0:
                    prediction_result = "احتمال الإصابة بمرض السكري منخفض."
                    prediction_class = "success"  # أخضر - Green
                else:
                    prediction_result = "احتمال الإصابة بمرض السكري مرتفع، يُفضل مراجعة الطبيب."
                    prediction_class = "warning"  # أحمر/برتقالي - Red/Orange
            else:
                prediction_result = "⚠️ النموذج غير متاح. يرجى تدريب النموذج أولاً."
                prediction_class = "danger"
                
        except Exception as e:
            prediction_result = f"⚠️ حدث خطأ: {str(e)}"
            prediction_class = "danger"
    
    return render_template('index.html', 
                         prediction=prediction_result,
                         prediction_class=prediction_class,
                         user_input=user_input)

# صفحة التحليل - Analysis Page
@app.route('/analysis')
def analysis():
    """
    عرض صفحة التحليل والرسوم البيانية
    Display analysis page with visualizations
    """
    # التحقق من وجود الرسوم البيانية - Check if plots exist
    plots = []
    plot_files = [
        'glucose_analysis.png',
        'age_analysis.png',
        'correlation_matrix.png',
        'model_comparison.png',
        'confusion_matrix.png'
    ]
    
    for plot_file in plot_files:
        if os.path.exists(f'static/{plot_file}'):
            plots.append(plot_file)
    
    if not plots:
        message = "⚠️ لا توجد رسوم بيانية. يرجى تشغيل train_model.py أولاً لإنشاء التحليلات."
    else:
        message = None
    
    return render_template('analysis.html', plots=plots, message=message)

# نقطة بداية التطبيق - Application entry point
if __name__ == '__main__':
    # التأكد من وجود المجلدات المطلوبة - Ensure required folders exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🩺 نظام توقع مرض السكري - Diabetes Prediction System")
    print("=" * 60)
    print("\n🌐 بدء تشغيل الخادم...")
    print("📍 افتح المتصفح على: http://0.0.0.0:5000")
    print("\n" + "=" * 60 + "\n")
    
    # تشغيل التطبيق - Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
