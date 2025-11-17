# pipe/ensure_gee_folders.py
# ============================================================
# Legt NUR diesen Ordner an:
#   projects/<project-id>/assets/<region_key>
# 
# und NICHT darüberliegende Ordner wie "projects" oder "<project-id>"
# ============================================================

import ee
import sys
from pathlib import Path

# Projekt-Root in sys.path setzen
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from bootstrap import init as bootstrap_init


def ensure_gee_region_folder(project_id: str, region_key: str, verbose=True) -> str:
    """
    Stellt sicher, dass NUR dieser Ordner erstellt wird:

        projects/<project-id>/assets/<region_key>

    (Die Ordner "projects", "<project-id>" und "<project-id>/assets" existieren IMMER.)
    """

    target = f"projects/{project_id}/assets/{region_key}"

    # Prüfen, ob der Ordner existiert
    try:
        ee.data.getAsset(target)
        if verbose:
            print(f"📂 Ordner existiert bereits: {target}")
        return target
    except Exception:
        pass  # muss erstellt werden

    # Ordner erstellen
    if verbose:
        print(f"📁 Erstelle GEE-Ordner: {target}")

    ee.data.createAsset({"type": "FOLDER"}, target)

    if verbose:
        print("✅ Ordner erstellt.")

    return target


# ------------------------------------------------------------
# CLI-Modus
# ------------------------------------------------------------
if __name__ == "__main__":
    cfg = bootstrap_init(verbose=True)

    project_id = cfg["gee"]["project_id"]
    region_key = cfg["defaults"]["region"]

    ensure_gee_region_folder(project_id, region_key, verbose=True)