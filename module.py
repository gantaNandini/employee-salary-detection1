import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
data = pd.read_csv("adult 3.csv")

# ---------------- SELECT FEATURES ----------------
features = ["age", "educational-num", "hours-per-week", "capital-gain", "capital-loss", "workclass", "occupation"]
target = "income"

# Handle missing values
data = data.replace("?", np.nan)
data = data.dropna(subset=features + [target])

# Encode categorical columns
label_encoders = {}
for col in ["workclass", "occupation"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode target variable
data[target] = data[target].apply(lambda x: 1 if ">50K" in x else 0)

# Split features & target
X = data[features]
y = data[target]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("✅ Model Training Completed!")
print(f"📌 Accuracy: {acc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model, scaler, and encoders
with open("salary_model.pkl", "wb") as file:
    pickle.dump({"model": model, "scaler": scaler, "label_encoders": label_encoders}, file)

print("💾 Model Saved Successfully!")
