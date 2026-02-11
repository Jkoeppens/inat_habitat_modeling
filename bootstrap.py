# ============================================================
# 🆕 bootstrap.py – neue Version (target/contrast-Kompatibilität)
# ============================================================

from pathlib import Path
import sys
from utils.yaml_loader import load_yaml_config
from utils.region import normalize_region
from utils.gee_init import initialize_gee

# Projektwurzel hinzufügen
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Projektwurzel bestimmen
# ------------------------------------------------------------
def get_project_root():
    return Path(__file__).resolve().parent


# ------------------------------------------------------------
# Target- und Kontrast-Spezies auswählen
# ------------------------------------------------------------
def select_species_pair(cfg, verbose=True):
    """
    Setzt cfg['selected_species'] = {'target': ..., 'contrast': ...}
    contrast kann 'all' sein → wird als Hintergrund klassifiziert.
    """
    defaults = cfg.get("defaults", {})
    all_species = cfg.get("species", {})

    target_key = defaults.get("target_species")
    contrast_key = defaults.get("contrast_species")

    # --- Target prüfen
    if not target_key:
        raise ValueError("❌ defaults.target_species fehlt.")
    if target_key not in all_species:
        raise ValueError(
            f"❌ Target-Spezies '{target_key}' nicht in species definiert.\n"
            f"   Verfügbar: {list(all_species.keys())}"
        )

    # --- Contrast prüfen
    if contrast_key != "all" and contrast_key not in all_species:
        raise ValueError(
            f"❌ Contrast-Spezies '{contrast_key}' nicht in species definiert "
            f"(oder 'all' ist nicht gesetzt).\n"
            f"   Verfügbar: {list(all_species.keys())} oder 'all'"
        )

    # --- Setzen
    cfg["selected_species"] = {
        "target": all_species[target_key],
        "contrast": ("background" if contrast_key == "all" else all_species[contrast_key])
    }

    if verbose:
        t_name = all_species[target_key]["name"]
        c_name = "background (all)" if contrast_key == "all" else all_species[contrast_key]["name"]
        print(f"🧬 Target-Species: {target_key} → {t_name}")
        print(f"🆚 Contrast-Species: {contrast_key} → {c_name}")

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
# species-spezifische Pfade erzeugen (nur für target)
# ------------------------------------------------------------
def apply_species_paths(cfg, verbose=True):
    base = Path(cfg["paths"]["base_data_dir"])
    species_key = cfg["defaults"]["target_species"]

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

    project_root = get_project_root()
    config_dir = project_root / "config"

    default_yaml = default_yaml or (config_dir / "default.yaml")
    local_yaml   = local_yaml   or (config_dir / "local.yaml")

    print(f"📄 default.yaml: {default_yaml}")
    print(f"📄 local.yaml:   {local_yaml}")

    # --- YAMLs laden (merge)
    cfg = load_yaml_config(default_yaml, local_yaml, verbose=verbose)

    # --- Region & Species-Paar auswählen
    select_region(cfg, verbose)
    select_species_pair(cfg, verbose)

    # --- Region normalisieren (UTM etc.)
    normalize_region(cfg, verbose=verbose)

    # --- Speziespfade
    apply_species_paths(cfg, verbose)

    # --- Earth Engine optional
    gee_project = cfg.get("gee", {}).get("project_id")
    if gee_project:
        print("\n🔧 Prüfe Earth Engine…")
        initialize_gee(gee_project, verbose=verbose)

    print("\n✅ BOOTSTRAP abgeschlossen.\n")
    return cfg