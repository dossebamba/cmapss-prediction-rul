"""Chargement des données brutes du dataset NASA C-MAPSS.

Convention de colonnes commune à tous les sous-ensembles (FD001-FD004) :
identifiant moteur, numéro de cycle, 3 réglages opérationnels, 21 mesures
capteurs.
"""

from pathlib import Path

import pandas as pd

COLUMN_NAMES = (
    ["unit_number", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SENSOR_COLUMNS = [c for c in COLUMN_NAMES if c.startswith("sensor_")]
SETTING_COLUMNS = [c for c in COLUMN_NAMES if c.startswith("op_setting_")]


def get_project_root() -> Path:
    """Racine du projet, déduite de l'emplacement de ce module (src/data/loaders.py)."""
    return Path(__file__).resolve().parents[2]


def _find_file(data_dir: Path, pattern: str) -> Path:
    """Cherche un fichier par motif dans data_dir, quel que soit le sous-dossier créé par l'extraction du zip Kaggle."""
    matches = list(data_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Fichier '{pattern}' introuvable dans {data_dir}. "
            "Vérifie que le dataset a été téléchargé (notebooks/00_ingestion)."
        )
    return matches[0]


def load_subset(subset: str, data_dir: Path | None = None):
    """Charge les fichiers train/test/RUL d'un sous-ensemble C-MAPSS (ex. 'FD001').

    Paramètres
    ----------
    subset : nom du sous-ensemble ('FD001', 'FD002', 'FD003' ou 'FD004')
    data_dir : dossier contenant les fichiers bruts (par défaut data/raw à la racine du projet)

    Retourne
    --------
    (df_train, df_test, df_rul) : tuple de DataFrames pandas
    """
    data_dir = data_dir or (get_project_root() / "data" / "raw")

    df_train = pd.read_csv(
        _find_file(data_dir, f"train_{subset}.txt"), sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    df_test = pd.read_csv(
        _find_file(data_dir, f"test_{subset}.txt"), sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    df_rul = pd.read_csv(
        _find_file(data_dir, f"RUL_{subset}.txt"), sep=r"\s+", header=None, names=["RUL"]
    )
    return df_train, df_test, df_rul
