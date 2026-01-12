from pathlib import Path
import json


HTML_TEMPLATE_PATH = Path("viz/html/main.html")
OUTPUT_PATH = Path("viz/surrogate_tree.html")

def export_html(tree_json_path):
    # JSON laden
    tree_data = json.loads(Path(tree_json_path).read_text(encoding="utf-8"))

    # HTML-Template laden
    template = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")

    # Platzhalter ersetzen
    html_out = template.replace("TREE_JSON", json.dumps(tree_data, ensure_ascii=False))

    # Speichern
    OUTPUT_PATH.write_text(html_out, encoding="utf-8")
    print(f"✓ HTML exportiert nach: {OUTPUT_PATH}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Pfad zur JSON-Datei des Surrogate-Trees")
    args = ap.parse_args()
    export_html(args.json)