# inat_habitat_modeling/utils/gee_init.py

import ee

def initialize_gee(project_id, verbose=True):
    """Robuste EE-Initialisierung."""

    if verbose:
        print("🔧 Prüfe Earth Engine...")

    # Test
    try:
        ee.Number(1).getInfo()
        if verbose:
            print("  ✔ EE bereits aktiv.")
        return True
    except:
        pass

    # Init
    try:
        ee.Initialize(project=project_id)
        if verbose:
            print("  ✔ EE initialisiert:", project_id)
        return True
    except:
        if verbose:
            print("  🔑 Versuche Auth...")

    # Auth
    try:
        ee.Authenticate()
        ee.Initialize(project=project_id)
        if verbose:
            print("  🔐 Auth erfolgreich.")
        return True
    except Exception as e:
        print("❌ EE-Fehler:", e)
        return False