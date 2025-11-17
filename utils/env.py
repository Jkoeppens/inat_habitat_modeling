# inat_habitat_modeling/utils/env.py

import subprocess
import sys
from pathlib import Path


# ----------------------------------------------------------------------
# 📦 Paketinstallation – robust & Notebook-tauglich
# ----------------------------------------------------------------------
def ensure_requirements(requirements_path: Path, quiet=True):
    """
    Installiert fehlende Pakete basierend auf requirements.txt.
    - quiet=True → kein Lärm im Notebook
    - quiet=False → pip-Ausgabe sichtbar
    """

    requirements_path = Path(requirements_path)

    if not requirements_path.exists():
        print(f"⚠️  requirements.txt nicht gefunden: {requirements_path}")
        return

    print(f"🔍 Prüfe Python-Pakete gemäß {requirements_path.name} ...")

    # stdout-Steuerung
    out = subprocess.PIPE if quiet else None

    try:
        # pip upgraden
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            stdout=out,
            stderr=out
        )

        # requirements installieren
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
            check=True,
            stdout=out,
            stderr=out
        )

        print("✅ Pakete installiert / aktualisiert.")
    except subprocess.CalledProcessError as e:
        print("❌ Fehler bei Paketinstallation:", e)