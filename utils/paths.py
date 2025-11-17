from pathlib import Path

def get_project_root() -> Path:
    """
    Gibt den Projektordner zurück:
    inat_habitat_modeling/
    """
    return Path(__file__).resolve().parents[1]