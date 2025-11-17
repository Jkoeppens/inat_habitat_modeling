# ============================================================
# bootstrap.py – zentrale Projektinitialisierung
# Klare, robuste, ordnerfeste Version
# ============================================================

from pathlib import Path
from utils.yaml_loader import load_yaml_config
from utils.region import normalize_region
from utils.gee_init import initialize_gee
import sys
from pathlib import Path

# Projektwurzel hinzufügen (Ordner, in dem bootstrap.py selbst liegt)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# Hilfsfunktion: Projektwurzel bestimmen
# ------------------------------------------------------------
def get_project_root():
    """
    bootstrap.py liegt im Projektordner:
        inat_habitat_modeling/
    Daher ist die Projektwurzel einfach das parent-Verzeichnis
    von dieser Datei.
    """
    return Path(__file__).resolve().parent


# ------------------------------------------------------------
# Spezies auswählen
# ------------------------------------------------------------
def select_species(cfg, verbose=True):
    species_key = cfg.get("defaults", {}).get("species")
    all_species = cfg.get("species", {})

    if not species_key:
        raise ValueError("❌ defaults.species fehlt.")

    if species_key not in all_species:
        raise ValueError(
            f"❌ Species '{species_key}' existiert nicht.\n"
            f"   Verfügbar: {list(all_species.keys())}"
        )

    cfg["selected_species"] = all_species[species_key]

    if verbose:
        print(f"🧬 Species: {species_key} → {cfg['selected_species']['name']}")

    return cfg


# ------------------------------------------------------------
# Region auswählen
# ------------------------------------------------------------
def select_region(cfg, verbose=True):
    region_key = cfg.get("defaults", {}).get("region")
    all_regions = cfg.get("regions", {})

    if not region_key:
        raise ValueError("❌ defaults.region fehlt.")

    if region_key not in all_regions:
        raise ValueError(
            f"❌ Region '{region_key}' existiert nicht.\n"
            f"   Verfügbar: {list(all_regions.keys())}"
        )

    cfg["region"] = all_regions[region_key]

    if verbose:
        print(f"🌍 Region: {region_key} → {cfg['region']['bbox_wgs84']}")

    return cfg


# ------------------------------------------------------------
# species-spezifische Pfade erzeugen
# ------------------------------------------------------------
def apply_species_paths(cfg, verbose=True):
    base = Path(cfg["paths"]["base_data_dir"])
    species_key = cfg["defaults"]["species"]

    cfg["paths"]["output_dir_species"]   = str(base / "outputs" / species_key)
    cfg["paths"]["features_dir_species"] = str(base / "features" / species_key)
    cfg["paths"]["temp_dir_species"]     = str(base / "temp" / species_key)

    if verbose:
        print("📂 Speziespfade:")
        for k in ["output_dir_species", "features_dir_species", "temp_dir_species"]:
            print(f"   • {k}: {cfg['paths'][k]}")

    return cfg


# ------------------------------------------------------------
# HAUPTFUNKTION: Projekt initialisieren
# ------------------------------------------------------------
def init(verbose=True, default_yaml=None, local_yaml=None):

    print("=========================================")
    print("🔧 BOOTSTRAP: Lade Konfiguration")
    print("=========================================")

    # 1) Projektwurzel korrekt bestimmen
    project_root = get_project_root()
    config_dir = project_root / "config"

    print(f"📁 Projektwurzel: {project_root}")

    # 2) YAML-Pfade setzen
    default_yaml = default_yaml or (config_dir / "default.yaml")
    local_yaml   = local_yaml   or (config_dir / "local.yaml")

    print(f"📄 default.yaml: {default_yaml}")
    print(f"📄 local.yaml:   {local_yaml}")

    # 3) YAMLs laden (deep merge + placeholder via yaml_loader)
    cfg = load_yaml_config(default_yaml, local_yaml, verbose=verbose)

    # 4) Region & Species auswählen
    select_region(cfg, verbose)
    select_species(cfg, verbose)

    # 5) Region normalisieren (UTM + bbox berechnen)
    normalize_region(cfg, verbose=verbose)

    # 6) Speziesabhängige Pfade
    apply_species_paths(cfg, verbose)

    # 7) Optional: Earth Engine
    gee_project = cfg.get("gee", {}).get("project_id")
    if gee_project:
        print("\n🔧 Prüfe Earth Engine…")
        initialize_gee(gee_project, verbose=verbose)

    print("\n✅ BOOTSTRAP abgeschlossen.\n")
    return cfg