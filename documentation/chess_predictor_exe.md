# Kurzanleitung: ChessPredictor.exe

Dieses Tool erkennt Schachstellungen aus Bildern und gibt sie als FEN-String oder Wahrscheinlichkeits-Matrix zurück.

---

## 1. Nutzung (Visuell)
Für einen schnellen Test mit grafischer Ausgabe.

1. Starte die `.exe` per Doppelklick.
2. Ziehe ein Bild in das schwarze Fenster und drücke `Enter`.
3. **Ergebnis:** Du siehst das Brett grafisch und den erkannten FEN-String.

---

## 2. Nutzung (Rohdaten / CLI)
Für die Integration in Skripte oder zur Weiterverarbeitung.

ChessPredictor.exe bild.jpg --raw
Optionale Flags:
--preprocessed: Wenn das Bild bereits exakt zugeschnitten ist.
--stdin: Um Bilddaten direkt als Bytes zu übergeben.

---

## 3. Beispiele:

Rohdaten statt Grafik (--raw):
```ChessPredictor.exe match.jpg --raw```

Bereits zugeschnittenes Bild (--preprocessed):
```ChessPredictor.exe dataset/crop_01.jpg --preprocessed --raw```

Powershell - Bilddaten direkt in die exe (--stdin):
```Get-Content bild.jpg -AsByteStream | ChessPredictor.exe --stdin --raw```

CMD - Bilddaten direkt in die exe (--stdin):
```type bild.jpg | ChessPredictor.exe --stdin --raw```

---

## 4. Format des Outputs:
[
  [0.01, 0.0, ..., 0.99],  // Feld 1
  [0.80, 0.1, ..., 0.05],  // Feld 2
  ...
  [0.0, 0.0, ..., 1.0]     // Feld 64
]

---

## 5. Mapping:

Index	Figur
(White pieces)
0	    Pawn
1	    Rook
2	    Knight
3	    Bishop
4	    Queen
5	    King
(Black pieces)
6	    Pawn
7	    Rook
8	    Knight
9	    Bishop
10	    Queen
11	    King
(No pieces)
12      Empty(.)