#!/usr/bin/env python3
"""
Lädt alle GEE-Export-Dateien aus Google Drive herunter und speichert sie lokal:

    /Volumes/Data/iNaturalist/data/raw/<region_key>

Benötigt:
    pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

from __future__ import annotations
import os
import io
import pathlib

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# ----------------------------------------------
# 🔧 GEE / Config
# ----------------------------------------------
import bootstrap


# ----------------------------------------------
# 🔧 Google Drive Settings
# ----------------------------------------------
DRIVE_FOLDER_NAME = "iNaturalist/data"   # GEE Export Ziel in Drive
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# ----------------------------------------------
# Google Drive Init
# ----------------------------------------------
def get_drive_service():
    """Authentifiziert die Google Drive API."""
    creds = None
    token_path = pathlib.Path("token_drive.json")
    creds_path = pathlib.Path("credentials_drive.json")

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_path.exists():
                raise RuntimeError(
                    "❌ credentials_drive.json fehlt! "
                    "Download über Google Cloud Console → OAuth 2.0 Client ID"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


# ----------------------------------------------
# Ordner in Google Drive finden
# ----------------------------------------------
def find_drive_folder_id(service, name):
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder'"
    resp = service.files().list(q=query, spaces="drive").execute()
    items = resp.get("files", [])
    if not items:
        raise RuntimeError(f"❌ Drive-Ordner '{name}' nicht gefunden.")
    return items[0]["id"]


def list_files(service, folder_id):
    query = f"'{folder_id}' in parents"
    resp = service.files().list(q=query, spaces="drive").execute()
    return resp.get("files", [])


# ----------------------------------------------
# Datei herunterladen
# ----------------------------------------------
def download_file(service, file_obj, target_dir):
    file_id = file_obj["id"]
    name = file_obj["name"]
    target = target_dir / name

    print(f"⬇️  Lade herunter: {name}")

    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(target, "wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"   Fortschritt: {int(status.progress() * 100)} %")

    print(f"   ✔ Gespeichert: {target}")


# ----------------------------------------------
# MAIN
# ----------------------------------------------
def main():
    print("=====================================")
    print("   📥 Lade GEE-Exports aus Drive")
    print("=====================================\n")

    # ------------------------------------------
    # ⚙️ Config laden → Region bestimmen
    # ------------------------------------------
    cfg = bootstrap.init(verbose=False)
    region_key = cfg["defaults"]["region"]

    # Lokaler Zielpfad:
    target_dir = pathlib.Path(f"/Volumes/Data/iNaturalist/data/raw/{region_key}")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Lokaler Zielpfad: {target_dir}")

    # ------------------------------------------
    # Google Drive Zugriff vorbereiten
    # ------------------------------------------
    service = get_drive_service()

    # Drive-Ordner-ID finden
    folder_id = find_drive_folder_id(service, DRIVE_FOLDER_NAME)

    # Dateien im Drive-Ordner auflisten
    files = list_files(service, folder_id)
    print(f"📦 {len(files)} Dateien gefunden.\n")

    # Alle herunterladen
    for f in files:
        download_file(service, f, target_dir)

    print("\n🎉 Download abgeschlossen.")
    print(f"   → Dateien liegen in {target_dir}")


if __name__ == "__main__":
    main()