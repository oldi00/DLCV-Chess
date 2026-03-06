"""Chess CV utilities adapted from CVChess with minor integration changes.

Attribution:
-----------
Source: Adapted from 'chess-cv'
URL: https://github.com/Luthiraa/CVChess/blob/main/src/final_notebook.ipynb
Authors: Luthira Abeykoon, Darshan Kasundra, Gawtham Senthilvelan, Ved Patel

Note:
-----
Includes minor modifications for integration; core functionality remains unchanged
from the original source.
"""

import cv2
import numpy as np

ID_TO_PIECE = {
    0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',
    6: 'p', 7: 'r', 8: 'n', 9: 'b', 10: 'q', 11: 'k',
    12: '1'  # empty
}

PIECE_TO_LABEL = {
    'P': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5,
    'p': 6, 'r': 7, 'n': 8, 'b': 9, 'q': 10, 'k': 11
}


def order_points_robust(pts):

    # First, get bounding box
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # sort by y-coord
    tl, bl = left_most[np.argsort(left_most[:, 1]), :]
    tr, br = right_most[np.argsort(right_most[:, 1]), :]

    return np.array([tl, tr, br, bl], dtype="float32")  # Clockwise


def warp_board(img, corners, output_size=400):

    src_pts = order_points_robust(corners)
    dst_pts = np.array([
        [0, 0],
        [output_size-1, 0],
        [output_size-1, output_size-1],
        [0, output_size-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (output_size, output_size))

    return warped


def pieces_to_fen(piece_list):

    board = [['1'] * 8 for _ in range(8)]

    def pos_to_index(pos):
        return (8 - int(pos[1]), ord(pos[0]) - ord('a'))

    for piece in piece_list:
        row, col = pos_to_index(piece['chessboard_position'])
        board[row][col] = ID_TO_PIECE[piece['category_id']]

    fen_rows = []
    for row in board:
        fen_row = ''
        empty_count = 0
        for cell in row:
            if cell == '1':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell

        if empty_count > 0:
            fen_row += str(empty_count)

        fen_rows.append(fen_row)

    return '/'.join(fen_rows)


def detect_board_corners(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_area = img.shape[0] * img.shape[1]

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.contourArea(approx) > 0.1 * image_area:
            corners = np.float32([pt[0] for pt in approx])
            ordered = order_points_robust(corners)
            return ordered, dilated

    return None, dilated
