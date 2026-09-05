# Prédiction de la Remaining Useful Life (RUL) — NASA C-MAPSS

Maintenance prédictive de moteurs d'avion à partir du dataset NASA C-MAPSS.

Ce projet prédit la Remaining Useful Life (RUL) de moteurs à turbine, le nombre de cycles de fonctionnement restants avant défaillance à partir de 21 séries temporelles de mesures capteurs. Deux stratégies de modélisation sont construites et comparées : une approche Machine Learning classique (features statistiques glissantes alimentant des modèles Random Forest et XGBoost) et une approche Deep Learning séquentielle (réseau LSTM sur des fenêtres de 30 cycles). Les modèles sont évalués via le RMSE, le MAE et le score NASA asymétrique, qui pénalise davantage les prédictions tardives (sur-estimation de la RUL) que les prédictions précoces (sous-estimation), conformément aux contraintes de sécurité industrielle réelles.

## Installation

```bash
git clone https://github.com/dossebamba/cmapss-prediction-rul.git
cd cmapss-prediction-rul
pip install -r requirements.txt
```

Le dataset est récupéré via l'API Kaggle ([behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)). Authentification requise :

1. Générer un token API sur [kaggle.com/settings](https://www.kaggle.com/settings) (section *API*)
2. Placer le fichier `kaggle.json` téléchargé dans `~/.kaggle/kaggle.json`

## Résultats

Comparaison finale sur le jeu de test (100 moteurs), après correction d'un biais de normalisation détecté en cours de projet (voir `notebooks/02_preparation/` et `notebooks/06_evaluation/`) :

| Modèle | RMSE (cycles) | MAE (cycles) | Score NASA (moyen) |
|---|---|---|---|
| Random Forest | 18,92 | 14,12 | 14,46 |
| XGBoost | 19,89 | 14,24 | 16,11 |
| **LSTM** | **13,19** | **9,51** | **2,89** |

Le **LSTM** est le modèle retenu : il devance nettement les deux autres, avec un écart qui se creuse fortement sur le score NASA (la métrique la plus représentative du risque opérationnel réel) — environ 5 fois moins de pénalité que Random Forest et XGBoost.

## Modèle final

Le modèle gagnant, son scaler de normalisation et sa configuration sont exportés dans `models/final/` — trois fichiers suffisants pour prédire une RUL à partir de mesures capteurs brutes, sans dépendre d'aucun notebook. Vérifié de façon autonome dans `notebooks/07_export/`.

## Structure du projet

- `data/` — brutes (`raw`), intermédiaires (`interim`) et prêtes pour la modélisation (`processed`), non versionnées
- `notebooks/` — pipeline complète numérotée, de l'ingestion (`00`) à l'export du modèle final (`07`)
- `src/` — fonctions réutilisables (chargement, feature engineering, modèles, métriques)
- `models/` — `ml/` et `dl/` : modèles d'entraînement/exploration (non versionnés) ; `final/` : modèle livrable, versionné
- `reports/` — figures et résultats de synthèse
- `config/` — paramètres centralisés (seeds, hyperparamètres, chemins)

## Utilisation

Exécuter les notebooks dans l'ordre de `notebooks/` (préfixes numériques), chaque étape produisant les entrées de la suivante. Pour utiliser directement le modèle final sans ré-exécuter toute la pipeline, voir `notebooks/07_export/07_export_modele_final.ipynb`.
