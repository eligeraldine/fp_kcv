import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("Student Stress Factors.csv")
df.drop("Timestamp", axis=1, inplace=True)
df.columns = [
    'Sleep_Quality', 'Headaches_per_week', 'Academic_Performance',
    'Study_Load', 'Extracurricular_per_week', 'Stress_Level'
]

# Label binarization
df['Stress_Label'] = df['Stress_Level'].apply(lambda x: 0 if x <= 2 else 1)
df.drop('Stress_Level', axis=1, inplace=True)

# Feature engineering
df['Load_Activity_Ratio'] = df['Study_Load'] / (df['Extracurricular_per_week'] + 1)

X = df.drop('Stress_Label', axis=1)
y = df['Stress_Label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
pickle.dump(model, open("stress_model_rf.sav", "wb"))
pickle.dump(scaler, open("stress_scaler.pkl", "wb"))

print("Model and scaler saved successfully!")
