📘 README.md — iNat Habitat Modeling

Ein modularer Workflow zur Habitat-Suitability-Modellierung auf Basis von:
	•	🐾 iNaturalist-Beobachtungen
	•	🛰️ Sentinel-2-Satellitendaten
	•	🌦️ Rasterisierter Klimatologie (monatsweise)
	•	🤖 XGBoost-Modellen für räumliche Vorhersagen

Ziel ist es, für eine gewählte Spezies und eine Kontrastklasse (z. B. ein häufiges Tier) robuste Habitat-Suitability-Maps zu erzeugen.

⸻

🔍 Inhalt
	1.	Features￼
	2.	Datenpipeline￼
	3.	Projektstruktur￼
	4.	Konfiguration￼
	5.	Training￼
	6.	Prediction Maps￼
	7.	Anforderungen￼
	8.	.gitignore￼

⸻

🚀 Features

✔ Automatische Projekterkennung und Bootstrapping
✔ Einheitliche Dateinamenkonventionen
✔ Modularisierte Feature-Pipeline
✔ Monatsbezogene Klimafeatures (NDVI, NDWI, Moran’s I, Geary’s C, Coverage)
✔ XGBoost-Training mit class-balanced weighting
✔ Feature Importance Export
✔ Habitat-Suitability-Maps als
	•	GeoTIFF
	•	PNG Preview
✔ Kachelweise Rasterverarbeitung (512×512)
✔ GPU/CPU neutral

⸻

🧬 Datenpipeline

1️⃣ Bootstrap

Definiert das Projekt:
	•	Region / Bounding Box
	•	Species
	•	Pfade
	•	Feature-Ordner
	•	Earth Engine (falls gebraucht)

2️⃣ Feature-Build

Erzeugt eine Feature-Tabelle:

inat_with_climatology_<species>_vs_<contrast>.csv

Format Beispiel:

| m07_ndvi_mean | m07_ndvi_std | m07_moran | … |

3️⃣ Training

XGBoost trainiert ein binäres Modell (1 = Zielart, 0 = Kontrastklasse):

model_<species>_vs_<contrast>.json
feature_importance_<species>_vs_<contrast>.png

ROC-AUC, Confusion Matrix etc. werden ausgegeben.

4️⃣ Habitat Prediction

Die Prediction-Engine generiert:

suitability_map_<species>_vs_<contrast>.tif
suitability_map_<species>_vs_<contrast>.png

mit dem zeitdurchschnittlichen Klimaraster (über alle Monate).

⸻

📁 Projektstruktur

inat_habitat_modeling/
│
├── config/
│   ├── default.yaml
│   └── local.yaml
│
├── pipe/
│   ├── build_features.py
│   ├── train_model.py
│   ├── make_prediction_map.py
│   └── utils/…
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── CLIMATOLOGY_MONTH_01.tif
│   │   ├── …
│   ├── outputs/
│   │   ├── feature_importance_*.png
│   │   ├── model_*.json
│   │   └── suitability_map_*.tif
│   └── features/
│
├── notebooks/
│   └── analysis.ipynb
│
└── README.md


⸻

⚙️ Konfiguration

config/default.yaml definiert:

region: berlin
species: macrolepiota_procera
contrast: parus_major
path:
  data_root: "/Volumes/Data/iNaturalist"

config/local.yaml überschreibt lokale Pfade oder Credentials.

⸻

🏋️‍♂️ Training

Im Notebook oder CLI:

python pipe/train_model.py

Ergebnis:
	•	model_<species>_vs_<contrast>.json
	•	feature_importance_<species>_vs_<contrast>.png
	•	ROC-AUC, confusion matrix, threshold

⸻

🗺️ Prediction Maps

Eine zeitlich aggregierte Habitat-Suitability-Map erzeugst du mit:

Notebook

!python pipe/make_prediction_map.py

CLI

python pipe/make_prediction_map.py

Output:

data/outputs/suitability_map_<species>_vs_<contrast>.tif
data/outputs/suitability_map_<species>_vs_<contrast>.png

Der aktuelle Ansatz verwendet durchschnittliche Klimaraster (über alle Monate) für eine reine räumliche Karte.

⸻

📦 Anforderungen

xgboost
rasterio
numpy
pandas
matplotlib
pyyaml
scikit-learn

Optional (für Bootstrapping):

earthengine-api


⸻

📄 .gitignore

Empfohlen:

# venv
venv/
*/__pycache__/

# data
data/raw/
data/processed/
data/outputs/
data/features/

# models
*.json
*.tif
*.png

# notebooks
.ipynb_checkpoints/

# OS
.DS_Store
