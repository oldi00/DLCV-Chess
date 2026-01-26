import json
import os
from collections import defaultdict

# ============================
# 1. Konfiguration
# ============================
ANNOTATIONS_PATH = (
    "G:/Meine Ablage/DLCV/annotations.json"  # Pfad zu Ihrer JSON anpassen!
)
NUM_SAMPLES = 5  # Wie viele Beispiele wollen Sie sehen?

# Mapping aus Ihrer preprocess.py
id_to_piece = {
    0: "P",
    1: "R",
    2: "N",
    3: "B",
    4: "Q",
    5: "K",
    6: "p",
    7: "r",
    8: "n",
    9: "b",
    10: "q",
    11: "k",
    12: "1",
}


def pieces_to_fen(piece_list):
    """
    Wandelt eine Liste von Figuren-Dicts in einen FEN-String um.
    """
    # 8x8 Brett mit '1' (leer) initialisieren
    board = [["1"] * 8 for _ in range(8)]
    pos_to_index = lambda pos: (8 - int(pos[1]), ord(pos[0]) - ord("a"))

    for piece in piece_list:
        # Sicherheitscheck, falls fehlerhafte Daten vorliegen
        if "chessboard_position" not in piece:
            continue

        row, col = pos_to_index(piece["chessboard_position"])

        # Mapping ID -> Buchstaben (z.B. 0 -> 'P')
        board[row][col] = id_to_piece[piece["category_id"]]

    # Das Brett in FEN-Reihen umwandeln
    fen_rows = []
    for row in board:
        fen_row = ""
        count = 0
        for cell in row:
            if cell == "1":
                fen_row += (
                    "0"  # Ihr Preprocess macht Nullen für leere Felder im String-Step?
                )
                count += 1
            else:
                if count > 0:
                    fen_row += str(count)
                    count = 0
                fen_row += cell

        if count > 0:
            fen_row += str(count)

        fen_rows.append(fen_row)

    return "/".join(fen_rows)


def inspect_annotations():
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"❌ Fehler: Datei nicht gefunden unter {ANNOTATIONS_PATH}")
        return

    print(f"📂 Lade Annotationen von: {ANNOTATIONS_PATH} ...")
    with open(ANNOTATIONS_PATH, "r") as f:
        data = json.load(f)

    # Dictionary erstellen: Image_ID -> Liste von Figuren
    # Das ist effizienter als jedes Mal durch die ganze Liste zu suchen
    image_to_pieces = defaultdict(list)
    for ann in data["annotations"]["pieces"]:
        image_to_pieces[ann["image_id"]].append(ann)

    # Die ersten X Bilder durchgehen
    images = data["images"][:NUM_SAMPLES]

    print(f"\n🔍 Zeige Ground Truth für die ersten {NUM_SAMPLES} Bilder:\n")

    for img in images:
        img_id = img["id"]
        file_name = img["path"]  # oder img['file_name'], je nach JSON Struktur

        # Figuren für dieses Bild holen
        pieces = image_to_pieces.get(img_id, [])

        # FEN generieren
        fen = pieces_to_fen(pieces)

        print("-" * 60)
        print(f"📷 Bild ID: {img_id}")
        print(f"📂 Datei:   {file_name}")
        print(f"♟️ Figuren: {len(pieces)}")
        print(f"🔑 FEN:     {fen}")
        print("-" * 60)


if __name__ == "__main__":
    inspect_annotations()
