# inat_habitat_modeling/utils/yaml_loader.py

import yaml
from pathlib import Path

# --------------------------------------------------------------
# Deep merge
# --------------------------------------------------------------
def deep_merge(a, b):
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a

# --------------------------------------------------------------
# Placeholder resolution
# --------------------------------------------------------------
def _find_all_keys(d, prefix=""):
    keys = []
    if isinstance(d, dict):
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else k
            keys.append(full)
            keys.extend(_find_all_keys(v, full))
    return keys

def _get_by_path(d, path):
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

def resolve_placeholders(cfg, root=None):
    if root is None:
        root = cfg

    if isinstance(cfg, dict):
        return {k: resolve_placeholders(v, root) for k, v in cfg.items()}

    if isinstance(cfg, list):
        return [resolve_placeholders(v, root) for v in cfg]

    if isinstance(cfg, str) and "${" in cfg:
        for key_path in _find_all_keys(root):
            ph = "${" + key_path + "}"
            if ph in cfg:
                val = _get_by_path(root, key_path)
                if val is not None:
                    cfg = cfg.replace(ph, str(val))
    return cfg

# --------------------------------------------------------------
# Load YAMLs, merge, resolve placeholders
# --------------------------------------------------------------
def load_yaml_config(default_path, local_path=None, verbose=True):
    default_path = Path(default_path)
    local_path = Path(local_path) if local_path else None

    if verbose:
        print("📄 Lade default.yaml:", default_path)

    with open(default_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if local_path and local_path.exists():
        if verbose:
            print("📄 Lade local.yaml:", local_path)
        with open(local_path, "r") as f:
            local = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, local)
        if verbose:
            print("  ✔ YAMLs gemerged.")

    cfg = resolve_placeholders(cfg)

    return cfg