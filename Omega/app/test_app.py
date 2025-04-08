import pytest
import joblib
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'vendor')

# Načtení modelu a encodery
model = joblib.load("../models/f1_winner_model.pkl")
enc_circuit = joblib.load("../encoders/enc_circuit.pkl")
enc_constructor = joblib.load("../encoders/enc_constructor.pkl")
enc_driver = joblib.load("../encoders/enc_driver.pkl")

def test_model_loaded():
    assert model is not None

def test_encoder_loaded():
    assert enc_circuit is not None
    assert enc_constructor is not None
    assert enc_driver is not None

def test_prediction_output():
    input_data = np.array([[2021, 5, 3, 2, 1]])  # umělé testovací hodnoty
    prob = model.predict_proba(input_data)[0][1]
    assert 0.0 <= prob <= 1.0

def test_valid_driver_combination():
    df = pd.read_csv("../f1_race_results.csv")
    row = df[
        (df["Season"] == 2021) &
        (df["Constructor"] == "Mercedes") &
        (df["Circuit"] == "Silverstone Circuit") &
        (df["Driver"] == "Hamilton")
    ]
    assert not row.empty
