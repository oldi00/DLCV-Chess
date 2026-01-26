import subprocess
import json
import numpy as np

process = subprocess.Popen(
    [
        r"C:/Users/olden/Desktop/programming/DLCV-Chess/dist/ChessPredictor.exe",
        r"G:/Meine Ablage/DLCV/ChessReD/images/0/G000_IMG001.jpg",
        "--raw",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
stdout, stderr = process.communicate()

raw_matrix = json.loads(stdout.strip())

predictions = []
mapping = {
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
    12: ".",
}

for feld_logits in raw_matrix:
    best_index = np.argmax(feld_logits)  # Finde Index der höchsten Zahl
    figur = mapping[best_index]
    predictions.append(figur)

board = np.array(predictions).reshape(8, 8)
# print(raw_matrix)
print()
for row in raw_matrix:
    print(row)
    print(len(row))
    print()
