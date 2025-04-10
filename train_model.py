import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# ========== LOAD DATA ==========
df = pd.read_csv("Student Stress Factors.csv")
df.drop('Timestamp', axis=1, inplace=True)
df.columns = [
    'Sleep_Quality', 'Headaches_per_week', 'Academic_Performance',
    'Study_Load', 'Extracurricular_per_week', 'Stress_Level'
]

# ========== PREPROCESSING ==========
df['Stress_Label'] = df['Stress_Level'].apply(lambda x: 0 if x <= 2 else 1)
df.drop('Stress_Level', axis=1, inplace=True)

# feature engineering
df['Load_Activity_Ratio'] = df['Study_Load'] / (df['Extracurricular_per_week'] + 1)

X = df.drop('Stress_Label', axis=1)
y = df['Stress_Label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== HANDLE IMBALANCED DATA ==========
# buat minority (not stress)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# ========== MODEL TRAINING ==========
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# cross-validation
# 3 fold means dilatih di 2 bagian, diuji di 1 bagian, terus diulang 3x dengan bagian yang berbeda sebagai test.

grid_search = GridSearchCV(model, param_grid, cv=3)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# ========== EVALUATION ==========
y_pred = best_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score

# Evaluasi training
y_train_pred = best_model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"\nTrain Accuracy: {train_acc:.4f}")

# Evaluasi testing (dengan angka akurasi) masi overfit deh :)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.4f}")


# ========== EXPORT MODEL & SCALER ==========
joblib.dump(best_model, 'best_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
