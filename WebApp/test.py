import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# === 1. Load Data ===

df = pd.read_csv("finaln.csv")

# === 2. Preprocess ===
X = df.drop("Severity", axis=1)
y = df["Severity"]

numeric_features = ['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Visibility(mi)']
categorical_features = [col for col in X.columns if col not in numeric_features]

# Scale numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numeric_features])
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_features)

# Combine with already numeric categorical features
X_final = pd.concat([X_scaled_df.reset_index(drop=True), X[categorical_features].reset_index(drop=True)], axis=1)

# === 3. Train Model ===
model = DecisionTreeClassifier(random_state=42)
model.fit(X_final, y)

# 



import pandas as pd
import joblib

# === Load data ===
df = pd.read_csv("finaln.csv")

# Filter rows with Severity == 2 or 3
filtered_df = df[df["Severity"].isin([2, 3])].copy()

# Drop target to simulate unseen data
X_new = filtered_df.drop("Severity", axis=1)

# Load model and scaler
model = joblib.load("decision_tree.pkl")
scaler = joblib.load("standard_scaler.pkl")

# Define numeric and categorical features
numeric_features = ['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Visibility(mi)']
categorical_features = [col for col in X_new.columns if col not in numeric_features]

# Scale numeric part
X_scaled = scaler.transform(X_new[numeric_features])
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_features)
print(X_scaled_df.head())

# Combine with other features
X_input = pd.concat([X_scaled_df.reset_index(drop=True), X_new[categorical_features].reset_index(drop=True)], axis=1)
print(X_input.head())

# Predict
predictions = model.predict(X_input)

# Optional: map severity descriptions
severity_mapping = {
    1: "Minor - No injuries, slight damage",
    2: "Moderate - Possible injuries, moderate damage",
    3: "Serious - Non-life-threatening injuries, significant damage",
    4: "Severe - Life-threatening injuries, major damage or fatality"
}

# Print predictions for first 5
for i, pred in enumerate(predictions[:5]):
    print(f"Sample {i+1} → Severity {pred}: {severity_mapping.get(pred)}")
