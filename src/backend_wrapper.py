import subprocess
import json
import os

# KONFIGURATION
# Pfad zu deiner kompilierten EXE (oder der Mock-EXE)
EXE_PATH = "dist/ChessPredictor.exe"
# Ein Testbild, das im selben Ordner liegen sollte
TEST_IMAGE = "G:/Meine Ablage/DLCV/ChessReD_Hough/0.png"


def predict_chess_board(image_path):
    # Prüfen, ob Bild existiert
    if not os.path.exists(image_path):
        print(f"[Backend] Fehler: Bild '{image_path}' nicht gefunden.")
        return None

    # Bild als Binärdaten (Bytes) einlesen
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"[Backend] Starte Inferenz mit {len(image_bytes)} Bytes...")

    try:
        # =========================================================
        # DEIN ANGEFORDERTER BLOCK
        # =========================================================
        # input=image_bytes   -> Schreibt die Bild-Daten direkt in 'sys.stdin' der EXE
        # capture_output=True -> Fängt auf, was die EXE 'print()'et (stdout)

        result = subprocess.run(
            [EXE_PATH], input=image_bytes, capture_output=True, timeout=10
        )
        # =========================================================

        # Prüfen, ob die EXE abgestürzt ist (Return Code != 0)
        if result.returncode != 0:
            print(f"[Backend] EXE fehlgeschlagen (Code {result.returncode})")
            # stderr dekodieren, um die Python-Fehlermeldung zu sehen
            err_msg = result.stderr.decode("utf-8", errors="replace")
            print(f"[Backend] Fehler-Log:\n{err_msg}")
            return None

        # Standard-Output (stdout) dekodieren
        json_str = result.stdout.decode("utf-8").strip()

        if not json_str:
            print("[Backend] Warnung: EXE hat nichts zurückgegeben.")
            return None

        # String in echtes Python-Objekt (Liste) umwandeln
        matrix = json.loads(json_str)

        return matrix

    except subprocess.TimeoutExpired:
        print("[Backend] Fehler: Timeout! Die EXE hat länger als 10s gebraucht.")
        return None
    except FileNotFoundError:
        print(f"[Backend] Fehler: Die Datei '{EXE_PATH}' wurde nicht gefunden.")
        return None
    except json.JSONDecodeError:
        print("[Backend] Fehler: Die Antwort der EXE war kein gültiges JSON.")
        print(f"Empfangen: {json_str[:100]}...")
        return None


if __name__ == "__main__":
    if not os.path.exists(TEST_IMAGE):
        print(f"Bitte lege ein Bild namens '{TEST_IMAGE}' in diesen Ordner.")
    else:
        probs = predict_chess_board(TEST_IMAGE)
        print(probs)
