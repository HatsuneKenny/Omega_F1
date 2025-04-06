# 📄 Dokumentace k projektu: Predikce vítěze závodu Formule 1 pomocí strojového učení

---

## 🧠 Popis projektu
Cílem projektu je vytvořit predikční model, který bude na základě reálných historických dat 
odhadovat pravděpodobnost výhry jezdce Formule 1. 
Projekt zahrnuje sběr dat, jejich zpracování, trénování modelu, vizualizace výsledků, a uživatelské rozhraní pomocí Streamlit.

---

## 🛠 Použité technologie
- Python 3
- Pandas, NumPy
- scikit-learn
- Streamlit
- Google Colab
- YAML (pro .conf soubor)

---

## 📦 Dataset
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

## 🧹 Předzpracování dat
- Výběr relevantních sloupců
- Label encoding pro textové proměnné
- Odstranění prázdných hodnot
- Výpočet binární proměnné `Winner`

---

## 🤖 Typ modelu
Použitý algoritmus: **Random Forest Classifier**  
Jedná se o **binární klasifikaci** – model předpovídá, zda jezdec vyhraje závod (1) nebo ne (0).

---

## 📈 Vyhodnocení modelu
- Přesnost modelu (accuracy score)
- Confusion Matrix
- Classification Report (precision, recall, F1-score)
- Vizualizace pomocí heatmap a grafů přesnosti dle sezóny

---

## 💻 Uživatelské rozhraní (Streamlit app)
- Název: `app.py`
- Výběr:
  - sezóny
  - okruhu
  - týmu
  - jezdce (filtruje se dle výběru)
  - startovní pozice
- Výstup: Pravděpodobnost výhry v procentech
- Ochrana před chybným vstupem: kontrola, zda daná kombinace existuje v datech
- Proč Streamlit a jak jsem na něj přišel: https://chatgpt.com/share/67efbd2e-def8-800c-82bc-c0b2f7705e7d
---

## ⚙️ Konfigurace aplikace
Konfigurační soubor `config.yaml`:
```yaml
# config.yaml

data_path: "f1_race_results.csv"
model_path: "f1_winner_model.pkl"
encoding_paths:
  circuit: "enc_circuit.pkl"
  constructor: "enc_constructor.pkl"
  driver: "enc_driver.pkl"
default_values:
  grid_position: 5
```
Používá se pro flexibilní načtení dat a modelu bez potřeby přepisovat kód.

---

## ▶ Spuštění

### Co je potřeba mít:
- `Python 3.8+`
- nainstalovaný modul `streamlit` (návod níže)
- soubory:
  - `streamlit_app.py`
  - `f1_race_results.csv`
  - `f1_winner_model.pkl`
  - `enc_circuit.pkl`
  - `enc_constructor.pkl`
  - `enc_driver.pkl`
  - `config.yaml`

### 🧪 Jak ověřit, že je nainstalován Streamlit:
```bash
streamlit --version
```

### ❌ Pokud není nainstalován:
```bash
pip install streamlit
```
**nebo (pokud nejsou admin práva):**
```bash
pip install --user streamlit
```

### ▶ Spuštění aplikace:
```bash
streamlit run streamlit_app.py
```
Aplikace se automaticky otevře v prohlížeči.

---

## 📁 Veškeré soubory
- `f1_race_results.csv`
- `train_model.py`
- `streamlit_app.py`
- `config.yaml`
- `f1_winner_model.pkl`
- `enc_circuit.pkl`, `enc_constructor.pkl`, `enc_driver.pkl`
- `f1_prediction_colab_vizualizace.ipynb` (notebook)
- `README.md` s návodem ke spuštění (volitelně)

---

## ✍️ Autor
**Filip Nejedlý**  
Projekt: Predikce vítěze závodu F1 pomocí strojového učení