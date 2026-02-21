import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ===============================
# STEP 1 — Load Dataset
# ===============================
print("Loading dataset...")
data = pd.read_csv("data/students.csv")

# ===============================
# STEP 2 — Split Features & Target
# ===============================
X = data.drop("risk_level", axis=1)
y = data["risk_level"]

# ===============================
# STEP 3 — Train Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# STEP 4 — Train Model
# ===============================
print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# STEP 5 — Evaluate Model
# ===============================
print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ===============================
# STEP 6 — Show Sample Probabilities
# ===============================
prob = model.predict_proba(X_test)

print("\nSample Risk Probabilities:")
print(prob[:5])

# ===============================
# STEP 7 — Save Model
# ===============================
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/student_model.pkl")

print("\nModel saved successfully at model/student_model.pkl")

print("\nFeature Importance:")

for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {round(importance, 3)}")