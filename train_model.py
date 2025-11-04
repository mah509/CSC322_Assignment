# -*- coding: utf-8 -*-
"""
نظام توقع مرض السكري - تدريب النموذج
Diabetes Prediction System - Model Training
"""

# استيراد المكتبات المطلوبة - Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# إعداد matplotlib للغة العربية - Setup matplotlib for Arabic
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("نظام توقع مرض السكري - Diabetes Prediction System")
print("=" * 60)

# تحميل البيانات - Load Pima Indians Diabetes Dataset
print("\n📊 تحميل بيانات مرض السكري...")
print("Loading Pima Indians Diabetes Dataset...")

# تحميل البيانات من مصدر موثوق - Load from reliable source
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

try:
    df = pd.read_csv(url, names=column_names)
    print("✅ تم تحميل البيانات بنجاح!")
    print(f"عدد الصفوف: {df.shape[0]}, عدد الأعمدة: {df.shape[1]}")
except:
    # في حالة عدم توفر الإنترنت، إنشاء بيانات تجريبية
    print("⚠️  إنشاء بيانات تجريبية...")
    np.random.seed(42)
    n_samples = 768
    df = pd.DataFrame({
        'Pregnancies': np.random.randint(0, 15, n_samples),
        'Glucose': np.random.randint(50, 200, n_samples),
        'BloodPressure': np.random.randint(40, 120, n_samples),
        'SkinThickness': np.random.randint(10, 60, n_samples),
        'Insulin': np.random.randint(0, 300, n_samples),
        'BMI': np.random.uniform(15, 50, n_samples),
        'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.5, n_samples),
        'Age': np.random.randint(21, 80, n_samples),
        'Outcome': np.random.randint(0, 2, n_samples)
    })

# عرض معلومات أساسية عن البيانات - Display basic information
print("\n📋 معاينة البيانات - Data Preview:")
print(df.head())
print("\n📈 الإحصاءات الوصفية - Descriptive Statistics:")
print(df.describe())

# التحقق من القيم المفقودة - Check for missing values
print("\n🔍 التحقق من القيم المفقودة:")
print(df.isnull().sum())

# استبدال القيم الصفرية غير الواقعية - Replace unrealistic zero values
# بعض الأعمدة لا يمكن أن تكون صفر (مثل الجلوكوز، ضغط الدم)
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_replace:
    df[col] = df[col].replace(0, df[col].median())

print("\n✅ تم معالجة القيم غير الواقعية")

# ═══════════════════════════════════════════════════════════
# التحليل الاستكشافي للبيانات - Exploratory Data Analysis (EDA)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("📊 بدء التحليل الاستكشافي للبيانات (EDA)")
print("=" * 60)

# إنشاء مجلد للرسوم البيانية - Create folder for plots
import os
if not os.path.exists('static'):
    os.makedirs('static')

# 1️⃣ العلاقة بين مستوى الجلوكوز والإصابة بالسكري
# Relationship between Glucose level and Diabetes
print("\n1️⃣  تحليل العلاقة بين مستوى الجلوكوز والإصابة بالسكري...")

plt.figure(figsize=(12, 5))

# رسم بياني 1: توزيع الجلوكوز حسب النتيجة
plt.subplot(1, 2, 1)
df[df['Outcome'] == 0]['Glucose'].hist(bins=30, alpha=0.7, label='No Diabetes', color='green', edgecolor='black')
df[df['Outcome'] == 1]['Glucose'].hist(bins=30, alpha=0.7, label='Diabetes', color='red', edgecolor='black')
plt.xlabel('Glucose Level (mg/dL)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Glucose Distribution by Diabetes Outcome', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# رسم بياني 2: صندوق الجلوكوز
plt.subplot(1, 2, 2)
df.boxplot(column='Glucose', by='Outcome', grid=False)
plt.xlabel('Diabetes Outcome (0=No, 1=Yes)', fontsize=12)
plt.ylabel('Glucose Level (mg/dL)', fontsize=12)
plt.title('Glucose Levels by Diabetes Status', fontsize=14, fontweight='bold')
plt.suptitle('')

plt.tight_layout()
plt.savefig('static/glucose_analysis.png', dpi=150, bbox_inches='tight')
print("✅ تم حفظ: static/glucose_analysis.png")
plt.close()

# 2️⃣ العلاقة بين العمر والإصابة بالسكري
# Relationship between Age and Diabetes
print("\n2️⃣  تحليل العلاقة بين العمر والإصابة بالسكري...")

plt.figure(figsize=(12, 5))

# رسم بياني 1: توزيع العمر حسب النتيجة
plt.subplot(1, 2, 1)
df[df['Outcome'] == 0]['Age'].hist(bins=20, alpha=0.7, label='No Diabetes', color='blue', edgecolor='black')
df[df['Outcome'] == 1]['Age'].hist(bins=20, alpha=0.7, label='Diabetes', color='orange', edgecolor='black')
plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Age Distribution by Diabetes Outcome', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# رسم بياني 2: صندوق العمر
plt.subplot(1, 2, 2)
df.boxplot(column='Age', by='Outcome', grid=False)
plt.xlabel('Diabetes Outcome (0=No, 1=Yes)', fontsize=12)
plt.ylabel('Age (years)', fontsize=12)
plt.title('Age by Diabetes Status', fontsize=14, fontweight='bold')
plt.suptitle('')

plt.tight_layout()
plt.savefig('static/age_analysis.png', dpi=150, bbox_inches='tight')
print("✅ تم حفظ: static/age_analysis.png")
plt.close()

# 3️⃣ مصفوفة الارتباط - Correlation Matrix
print("\n3️⃣  إنشاء مصفوفة الارتباط...")

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Feature Relationships', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('static/correlation_matrix.png', dpi=150, bbox_inches='tight')
print("✅ تم حفظ: static/correlation_matrix.png")
plt.close()

# ═══════════════════════════════════════════════════════════
# تحضير البيانات للتدريب - Prepare Data for Training
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("⚙️  تحضير البيانات للتدريب")
print("=" * 60)

# فصل المتغيرات المستقلة والتابعة - Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"\n✅ عدد العينات: {X.shape[0]}")
print(f"✅ عدد المتغيرات: {X.shape[1]}")
print(f"✅ توزيع النتائج:")
print(f"   - غير مصابين (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"   - مصابين (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")

# تقسيم البيانات - Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ بيانات التدريب: {X_train.shape[0]} عينة")
print(f"✅ بيانات الاختبار: {X_test.shape[0]} عينة")

# تطبيع البيانات - Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# حفظ المطبع للاستخدام لاحقاً - Save scaler for later use
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ تم حفظ المطبع (Scaler): scaler.pkl")

# ═══════════════════════════════════════════════════════════
# تدريب النماذج - Train Machine Learning Models
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("🤖 تدريب نماذج التعلم الآلي")
print("=" * 60)

# قاموس لحفظ النماذج والنتائج - Dictionary to store models and results
models = {}
results = {}

# 1️⃣ نموذج الانحدار اللوجستي - Logistic Regression
print("\n1️⃣  تدريب نموذج الانحدار اللوجستي (Logistic Regression)...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
models['Logistic Regression'] = lr_model
results['Logistic Regression'] = lr_accuracy
print(f"   ✅ دقة النموذج: {lr_accuracy * 100:.2f}%")

# 2️⃣ نموذج الغابة العشوائية - Random Forest
print("\n2️⃣  تدريب نموذج الغابة العشوائية (Random Forest)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
models['Random Forest'] = rf_model
results['Random Forest'] = rf_accuracy
print(f"   ✅ دقة النموذج: {rf_accuracy * 100:.2f}%")

# 3️⃣ نموذج XGBoost
print("\n3️⃣  تدريب نموذج XGBoost...")
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
models['XGBoost'] = xgb_model
results['XGBoost'] = xgb_accuracy
print(f"   ✅ دقة النموذج: {xgb_accuracy * 100:.2f}%")

# ═══════════════════════════════════════════════════════════
# مقارنة النماذج - Compare Models
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("📊 مقارنة أداء النماذج")
print("=" * 60)

# طباعة النتائج - Print results
print("\n🏆 نتائج دقة النماذج:")
print("-" * 40)
for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:20s}: {accuracy * 100:6.2f}%")

# اختيار أفضل نموذج - Select best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_accuracy = results[best_model_name]

print("\n" + "=" * 60)
print(f"🥇 أفضل نموذج: {best_model_name}")
print(f"🎯 الدقة: {best_accuracy * 100:.2f}%")
print("=" * 60)

# حفظ أفضل نموذج - Save best model
joblib.dump(best_model, 'diabetes_model.pkl')
print(f"\n✅ تم حفظ النموذج: diabetes_model.pkl")

# رسم بياني لمقارنة النماذج - Plot model comparison
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[m] * 100 for m in model_names]
colors = ['#3498db', '#2ecc71', '#e74c3c']

bars = plt.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Model Name', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, 100])
plt.grid(axis='y', alpha=0.3)

# إضافة القيم على الأعمدة - Add values on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('static/model_comparison.png', dpi=150, bbox_inches='tight')
print("✅ تم حفظ: static/model_comparison.png")
plt.close()

# ═══════════════════════════════════════════════════════════
# تقرير التصنيف النهائي - Final Classification Report
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(f"📋 تقرير التصنيف التفصيلي - {best_model_name}")
print("=" * 60)

# الحصول على تنبؤات أفضل نموذج - Get predictions from best model
if best_model_name == 'Logistic Regression':
    best_pred = lr_pred
elif best_model_name == 'Random Forest':
    best_pred = rf_pred
else:
    best_pred = xgb_pred

print("\n" + classification_report(y_test, best_pred, 
                                    target_names=['No Diabetes', 'Diabetes']))

# مصفوفة الالتباس - Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\n✅ تم حفظ: static/confusion_matrix.png")
plt.close()

print("\n" + "=" * 60)
print("✅ اكتمل التدريب بنجاح!")
print("=" * 60)
print("\n📦 الملفات المحفوظة:")
print("   - diabetes_model.pkl (النموذج المدرب)")
print("   - scaler.pkl (المطبع)")
print("   - static/glucose_analysis.png")
print("   - static/age_analysis.png")
print("   - static/correlation_matrix.png")
print("   - static/model_comparison.png")
print("   - static/confusion_matrix.png")
print("\n🚀 جاهز لتشغيل التطبيق!")
