import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
import sys
sys.path.insert(0, 'vendor')

# Načtení datasetu
df = pd.read_csv("f1_race_results.csv")

# Cílová proměnná
df["Final Position"] = df["Final Position"].astype(int)
df["Winner"] = (df["Final Position"] == 1).astype(int)

# Label encoding
encoder_circuit = LabelEncoder()
encoder_constructor = LabelEncoder()
encoder_driver = LabelEncoder()

df["Circuit"] = encoder_circuit.fit_transform(df["Circuit"])
df["Constructor"] = encoder_constructor.fit_transform(df["Constructor"])
df["Driver"] = encoder_driver.fit_transform(df["Driver"])

# Výběr atributů
features = ["Season", "Circuit", "Grid Position", "Constructor", "Driver"]
X = df[features]
y = df["Winner"]

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trénink modelu
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Uložení modelu a encoderů
os.makedirs("models", exist_ok=True)
os.makedirs("encoders", exist_ok=True)

joblib.dump(model, "models/f1_winner_model.pkl")
joblib.dump(encoder_circuit, "encoders/enc_circuit.pkl")
joblib.dump(encoder_constructor, "encoders/enc_constructor.pkl")
joblib.dump(encoder_driver, "encoders/enc_driver.pkl")

# Vyhodnocení
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Přesnost modelu: {accuracy * 100:.2f}%")
