# Prédiction de la Remaining Useful Life (RUL) — NASA C-MAPSS

Maintenance prédictive de moteurs d'avion à partir du dataset NASA C-MAPSS.

Ce projet prédit la Remaining Useful Life (RUL) de moteurs à turbine, le nombre de cycles de fonctionnement restants avant défaillance à partir de 21 séries temporelles de mesures capteurs. Deux stratégies de modélisation sont construites et comparées : une approche Machine Learning classique (features statistiques glissantes alimentant des modèles Random Forest et XGBoost) et une approche Deep Learning séquentielle (réseau LSTM sur des fenêtres de 30 cycles). Les modèles sont évalués via le RMSE, le MAE et le score NASA asymétrique, qui pénalise davantage les prédictions tardives (sous-estimation de la RUL) que les prédictions précoces, conformément aux contraintes de sécurité industrielle réelles.

## Installation

```bash
git clone <url-du-repo>
cd cmapss-prediction-rul
pip install -r requirements.txt
```

Le dataset est récupéré via l'API Kaggle ([behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)). Authentification requise :

1. Générer un token API sur [kaggle.com/settings](https://www.kaggle.com/settings) (section *API*)
2. Placer le fichier `kaggle.json` téléchargé dans `~/.kaggle/kaggle.json`

## Structure du projet

- `data/` — brutes (`raw`), intermédiaires (`interim`) et prêtes pour la modélisation (`processed`), non versionnées
- `notebooks/` — pipeline complète, numérotée de l'ingestion à l'évaluation
- `src/` — fonctions réutilisables (chargement, feature engineering, modèles, métriques)
- `models/` — modèles entraînés sérialisés
- `reports/` — figures et résultats de synthèse
- `config/` — paramètres centralisés (seeds, hyperparamètres, chemins)

## Utilisation

Exécuter les notebooks dans l'ordre de `notebooks/` (préfixes numériques), chaque étape produisant les entrées de la suivante.
