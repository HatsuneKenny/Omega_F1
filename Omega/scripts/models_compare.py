import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("../f1_race_results.csv")
df["Winner"] = (df["Final Position"] == 1).astype(int)

# Load encoders
enc_circuit_rf = joblib.load("../encoders/enc_circuit.pkl")
enc_constructor_rf = joblib.load("../encoders/enc_constructor.pkl")
enc_driver_rf = joblib.load("../encoders/enc_driver.pkl")

enc_circuit_mlp = joblib.load("../encoders/enc_mlp_circuit.pkl")
enc_constructor_mlp = joblib.load("../encoders/enc_mlp_constructor.pkl")
enc_driver_mlp = joblib.load("../encoders/enc_mlp_driver.pkl")

# Encode data for both models
df_rf = df.copy()
df_mlp = df.copy()

# RF encodings
df_rf["Circuit"] = enc_circuit_rf.transform(df_rf["Circuit"])
df_rf["Constructor"] = enc_constructor_rf.transform(df_rf["Constructor"])
df_rf["Driver"] = enc_driver_rf.transform(df_rf["Driver"])

# MLP encodings
df_mlp["Circuit"] = enc_circuit_mlp.transform(df_mlp["Circuit"])
df_mlp["Constructor"] = enc_constructor_mlp.transform(df_mlp["Constructor"])
df_mlp["Driver"] = enc_driver_mlp.transform(df_mlp["Driver"])

# Features and target
features = ["Season", "Circuit", "Grid Position", "Constructor", "Driver"]
X_rf = df_rf[features]
X_mlp = df_mlp[features]
y = df["Winner"]

# Load models
rf_model = joblib.load("../models/f1_winner_model.pkl")
mlp_model = joblib.load("../models/f1_mlp_model.pkl")

# Predict
y_pred_rf = rf_model.predict(X_rf)
y_pred_mlp = mlp_model.predict(X_mlp)

# Evaluate
print("\n🌟 Random Forest")
print("Accuracy:", accuracy_score(y, y_pred_rf))
print(classification_report(y, y_pred_rf))

print("\n🤖 Neuronka (MLP)")
print("Accuracy:", accuracy_score(y, y_pred_mlp))
print(classification_report(y, y_pred_mlp))