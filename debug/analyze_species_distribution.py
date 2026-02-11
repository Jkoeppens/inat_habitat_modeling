#!/usr/bin/env python3

"""
analyze_species_merge.py

Visualisiert die Verteilung von Beobachtungen in einer gemergten iNat/GBIF-CSV.

Zeigt:
- Jahresverlauf
- Monatsverlauf
- Räumliche Verteilung

Standardmäßig:
- zieht Ziel- und Kontrastart aus der Konfiguration
- verwendet Pfadkonvention: outputs/<slug>/inat_merged_<slug>_vs_<slug>.csv

Optional:
- per --target und --contrast überschreibbar
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Projekt-Root finden und zum Pfad hinzufügen
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import init as bootstrap_init

sns.set(style="whitegrid")


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def slug(name: str) -> str:
    """Macht einen slug-artigen Dateinamen (lowercase, underscores)."""
    return name.replace(" ", "_").lower()


def load_config():
    return bootstrap_init(verbose=False)


def get_csv_path(cfg, target_name, contrast_name):
    """Sucht CSV-Datei basierend auf Konvention."""
    out_root = Path(cfg["paths"]["output_dir"])
    tslug = slug(target_name)
    cslug = slug(contrast_name)

    species_dir = out_root / tslug
    fname = f"inat_merged_{tslug}_vs_{cslug}.csv"
    direct = species_dir / fname
    fallback = out_root / fname

    if direct.exists():
        return direct
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"❌ Merge-CSV nicht gefunden: {direct} oder {fallback}")


def plot_and_show(fig):
    """Zeigt oder speichert ein matplotlib-Figure-Objekt."""
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def analyze(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ Merge-CSV nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path)
    print("✅ CSV geladen:", csv_path)
    print("🔢 Zeilen:", len(df))
    print("\n📊 Species-Verteilung:")
    print(df["species"].value_counts())
    print("\n📊 Label-Verteilung:")
    print(df["label"].value_counts())

    out_dir = csv_path.parent
    tslug = csv_path.stem.split("_")[2]

    # 📆 Jahresverlauf
    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=df,
        x="year",
        hue="label",
        bins=range(df["year"].min(), df["year"].max() + 2),
        multiple="stack",
        palette={0: "lightgray", 1: "tab:blue"}
    )
    plt.title("📆 Jährlicher Verlauf")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tslug}_yearly.png")
    plt.close()

    # 📅 Monatsverlauf
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month

    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=df,
        x="month",
        hue="label",
        bins=12,
        multiple="stack",
        palette={0: "lightgray", 1: "tab:blue"}
    )
    plt.title("📅 Monatlicher Verlauf")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tslug}_monthly.png")
    plt.close()

    # 🌍 Räumliche Verteilung
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=df,
        x="longitude",
        y="latitude",
        hue="label",
        palette={0: "lightgray", 1: "tab:blue"},
        alpha=0.5,
        s=10,
        edgecolor=None
    )
    plt.title("🌍 Räumliche Verteilung")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tslug}_spatial.png")
    plt.close()

    print("✅ Plots gespeichert in:", out_dir)


# ------------------------------------------------------------
# CLI-Wrapper
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="Name der Zielart", default=None)
    parser.add_argument("--contrast", help="Name der Kontrastart", default=None)
    args = parser.parse_args()

    cfg = load_config()

    # Ziel- und Kontrastart aus der Konfig oder CLI
    target_key = cfg["defaults"]["target_species"]
    contrast_key = cfg["defaults"].get("contrast_species", "all")

    target = args.target or cfg["species"][target_key]["name"]
    contrast = args.contrast or cfg["species"][contrast_key]["name"]

    print(f"\n🎯 Zielart: {target}")
    print(f"⚔️  Kontrastart: {contrast}")

    # CSV suchen & analysieren
    csv_path = get_csv_path(cfg, target, contrast)
    analyze(csv_path)


if __name__ == "__main__":
    main()