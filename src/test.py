import subprocess
import os
import json
import sys
import numpy as np


# Konfiguration
EXE_PATH = "/dist/ChessPredictor.exe"  # Pfad zu Ihrer neuen EXE
IMAGE_PATH = (
    r"G:/Meine Ablage/DLCV/ChessReD/images/0/G000_IMG000.jpg"  # Pfad zu einem Testbild
)


def run_prediction_with_bytes(exe_path, image_path):
    """
    Liest ein Bild als Bytes ein, sendet es an die EXE und holt das JSON-Ergebnis.
    """

    # 1. Prüfen, ob Dateien existieren
    if not os.path.exists(exe_path):
        print(f"❌ Fehler: EXE nicht gefunden unter {exe_path}")
        return
    if not os.path.exists(image_path):
        print(f"❌ Fehler: Bild nicht gefunden unter {image_path}")
        return

    print(f"📤 Lade Bild als Bytes: {image_path}...")

    # 2. Das Bild als reine Bytes lesen ('rb')
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"🚀 Sende {len(image_bytes)} Bytes an die EXE...")

    # 3. Den Prozess starten und Bytes via STDIN übergeben
    # Flags: --stdin (für Byte-Input) und --raw (für JSON-Output)
    try:
        process = subprocess.run(
            [exe_path, "--stdin", "--raw"],
            input=image_bytes,  # Hier fließen die Bytes rein
            capture_output=True,  # Wir wollen die Antwort abfangen
            text=False,  # Wichtig: Wir empfangen rohe Bytes/Text gemischt, aber Python soll stdout dekodieren
        )

        # Den Output manuell dekodieren
        stdout_str = process.stdout.decode("utf-8").strip()
        stderr_str = process.stderr.decode("utf-8").strip()

        if process.returncode != 0:
            print("\n❌ Die EXE hat einen Fehler gemeldet:")
            print(stderr_str)
            return

        # 4. Ergebnis verarbeiten
        print("\n✅ Empfangene Daten von EXE:")

        # Versuch, das JSON zu parsen
        try:
            # Manchmal gibt es Warnungen vor dem JSON, wir suchen die Klammern
            json_start = stdout_str.find("[[")
            json_end = stdout_str.rfind("]]") + 2

            if json_start != -1 and json_end != -1:
                clean_json = stdout_str[json_start:json_end]
                data = json.loads(clean_json)

                print(f"   Daten-Struktur: {len(data)} Zeilen x {len(data[0])} Spalten")
                print(
                    f"   Erster Wert (A1): {np.sum(data[0])}"
                )  # Beispiel-Logits für Feld A1
            else:
                print("   Kein gültiges JSON gefunden. Roh-Output:")
                print(stdout_str)

        except json.JSONDecodeError:
            print("   Konnte JSON nicht lesen. Roh-Output:")
            print(stdout_str)

    except Exception as e:
        print(f"❌ Ein unerwarteter Fehler ist aufgetreten: {e}")


if __name__ == "__main__":
    # Wenn Sie das Skript mit einem Argument aufrufen: python test.py bild.jpg
    # Sonst nimmt es den Standard-Pfad oben.
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]

    run_prediction_with_bytes(EXE_PATH, IMAGE_PATH)
