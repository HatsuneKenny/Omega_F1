#  Dokumentace k projektu: Predikce vítěze závodu Formule 1 pomocí strojového učení

---

## Popis projektu
Cílem projektu je vytvořit predikční model, který bude na základě reálných historických dat 
odhadovat pravděpodobnost výhry jezdce Formule 1. 
Projekt zahrnuje sběr dat, jejich zpracování, trénování modelu, vizualizace výsledků, a uživatelské rozhraní pomocí Streamlit.

---

## Použité technologie
- Python 3
- Pandas, NumPy
- scikit-learn
- Streamlit
- Google Colab
- YAML (pro .conf soubor)

---

## Dataset
- Název souboru: `f1_race_results.csv`
- Počet záznamů: cca 9600(povinný počet 1500+)
- Atributy:
  - `Season`
  - `Circuit`
  - `Grid Position`
  - `Constructor`
  - `Driver`
  - `Final Position`
- Cílová proměnná: `Winner` (1 = vyhrál, 0 = nevyhrál)

---

## Předzpracování dat
- Výběr relevantních sloupců
- Label encoding pro textové proměnné
- Odstranění prázdných hodnot
- Výpočet binární proměnné `Winner`

---

## Typy modelů
Použité algoritmy:
- **Random Forest Classifier** – binární klasifikace
- **Multi-Layer Perceptron (Neuronová síť)** – také klasifikace

Oba modely lze mezi sebou porovnat pomocí `compare_models.py` a vybrat ve Streamlit aplikaci.

---

## Vyhodnocení modelu
- Accuracy score (přesnost)
- R² score (koeficient determinace)
- Classification report (precision, recall, F1-score)
- Confusion Matrix (vizualizace pomocí seaborn)

---

## Uživatelské rozhraní (Streamlit app)
- Název: `main.py`
- Možnosti výběru:
  - model (Random Forest / Neuronová síť)
  - sezóna
  - okruh
  - tým
  - jezdec (filtruje se dle týmu a sezóny)
  - startovní pozice
- Ochrana vstupu: kontroluje, zda daná kombinace existuje v datech
- Výsledek: pravděpodobnost výhry v %

---

## Konfigurace aplikace
Konfigurační soubor `config.yaml`:
```yaml
# config.yaml

data_path: "f1_race_results.csv"

model_paths:
  random_forest: "models/f1_winner_model.pkl"
  neural_network: "models/f1_mlp_model.pkl"

encoding_paths:
  random_forest:
    circuit: "encoders/enc_circuit.pkl"
    constructor: "encoders/enc_constructor.pkl"
    driver: "encoders/enc_driver.pkl"

  neural_network:
    circuit: "encoders/enc_mlp_circuit.pkl"
    constructor: "encoders/enc_mlp_constructor.pkl"
    driver: "encoders/enc_mlp_driver.pkl"

default_values:
  grid_position: 5
```

---

## ▶ Instalace a spuštění

###  Instalace knihoven
Projekt používá soubor `requirements.txt`, kde jsou uvedeny všechny potřebné knihovny.

#### 1. Vytvoření prostředí (volitelne):
```bash
python -m venv .venv
```

#### 2. Aktivace prostředí:
#####  macOS/Linux:
```bash
source .venv/bin/activate
```
#####  Windows:
```bash
.venv\Scripts\activate
```

#### 3. Instalace knihoven:
```bash
pip install -r requirements.txt
```

---

### Spuštění aplikace:
```bash
streamlit run main.py

cd /Users/filipnejedly/Documents/Python/Omega
streamlit run app/main.py

```
Aplikace se automaticky otevře v prohlížeči.

---

##  Struktura projektu
```
Omega/
├── app/
│   └── main.py
├── models/
│   ├── f1_winner_model.pkl
│   └── f1_mlp_model.pkl
├── encoders/
│   └── *.pkl
├── notebooks/
│   ├── F1_Omega_MLP_colab.ipynb
│   └── F1_Omega_RNDFORST_colab.ipynb
├── scripts/
│   ├── train_model.py
│   ├── train_mlp_model.py
│   └── compare_models.py
├── f1_race_results.csv
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Autor
**Filip Nejedlý**  
Projekt: Predikce vítěze závodu F1 pomocí strojového učení
