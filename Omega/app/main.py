import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yaml
import os
import logging
import sys
from sklearn.ensemble import RandomForestClassifier

#  Zde loggujes hnupe
logging.basicConfig(
    filename='f1_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#  Tady se nacita confing soubor(config.yaml)
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data_path"]
MODEL_PATHS = config["model_paths"]
ENCODING_PATHS_ALL = config["encoding_paths"]
DEFAULT_GRID = config["default_values"]["grid_position"]

#  Tu vytvaris hlavicku Streamlitu magore(jmeno stranky, layout, nadpis, apod...)
st.set_page_config(page_title="F1 Winner Predictor", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #ff4b4b;'>🏎️ F1 Winner Predictor</h1>
    <p style='text-align: center; color: gray;'>Zjisti pravděpodobnost výhry jezdce podle historických dat</p>
""", unsafe_allow_html=True)

#  Kontrola jestli dataset vubec existuje(jestli neni treba smazany)
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error("Nepodařilo se načíst dataset.") # pokud nebyl nalezen vyhodi se chyba
    logging.error(f"Dataset error: {e}")
    st.stop()

available_seasons = sorted(df["Season"].unique())
circuit_names = df["Circuit"].unique()
constructor_names = df["Constructor"].unique()

#  Tady se uzivatel vybere mezi modelama(RF a MLP). Ke kazdemu modelu jsou prirazene encodery
model_choice = st.radio("Vyber model:", ["Random Forest", "Neuronová síť"])
model_key = "random_forest" if model_choice == "Random Forest" else "neural_network"
model_path = MODEL_PATHS[model_key]
encoding_paths = ENCODING_PATHS_ALL[model_key]

#  Nacteni modelu a encoderu zda existuji(jestli ne vyhodi se chyba)
try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Soubor modelu {model_path} nenalezen.")
    model = joblib.load(model_path)

    encoder_circuit = joblib.load(encoding_paths["circuit"])
    encoder_constructor = joblib.load(encoding_paths["constructor"])
    encoder_driver = joblib.load(encoding_paths["driver"])
except Exception as e:
    logging.error(f"Model nebo encodery se nepodařilo načíst: {e}")
    st.error("Model nebo encodery se nepodařilo načíst. Zkontroluj soubory a konfiguraci.")
    st.stop()

#  Tady se vytvari UI
st.markdown("---")
st.markdown("### 📊 Zadej parametry závodu")

col1, col2 = st.columns(2)
with col1:
    season = st.selectbox("Sezóna", available_seasons)
    constructor = st.selectbox("Tým", constructor_names)
with col2:
    circuit = st.selectbox("Okruh", circuit_names)
    grid_pos = st.number_input("Startovní pozice", min_value=1, max_value=30, value=DEFAULT_GRID)

# Kontrolo toho aby uzivatele nezadal blbost
df_raw = pd.read_csv(DATA_PATH)
filtered = df_raw[
    (df_raw["Season"] == season) &
    (df_raw["Constructor"] == constructor) &
    (df_raw["Circuit"] == circuit)
]
driver_names = sorted(filtered["Driver"].unique())

if len(driver_names) == 0:
    st.error("⚠️ Pro tuto kombinaci sezóny, týmu a okruhu nejsou dostupní žádní jezdci.")
    st.stop()

driver = st.selectbox("Jezdec", driver_names)

st.markdown("---")

if st.button("🔬 Spustit predikci"):
    row_exists = df_raw[
        (df_raw["Season"] == season) &
        (df_raw["Circuit"] == circuit) &
        (df_raw["Constructor"] == constructor) &
        (df_raw["Driver"] == driver)
    ].shape[0] > 0

    if not row_exists:
        st.error("⚠️ Tato kombinace sezóny, okruhu, týmu a jezdce neexistuje v datech.")
    else:
        circuit_encoded = encoder_circuit.transform([circuit])[0]
        constructor_encoded = encoder_constructor.transform([constructor])[0]
        driver_encoded = encoder_driver.transform([driver])[0]

        input_data = np.array([[season, circuit_encoded, grid_pos, constructor_encoded, driver_encoded]])
        probability = model.predict_proba(input_data)[0][1]

        logging.info(
            f"[{model_choice}] Predikce: season={season}, circuit={circuit}, team={constructor}, driver={driver}, grid={grid_pos}, result={probability:.4f}"
        )

        st.markdown("---")
        st.metric(label=f"🚗 Šance na výhru pro {driver}", value=f"{probability * 100:.2f}%")
        st.success(f"Predikce pomocí **{model_choice}** proběhla úspěšně pro závod v **{circuit}**, sezóna **{season}**.")
