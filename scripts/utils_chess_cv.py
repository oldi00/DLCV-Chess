"""
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

import numpy as np
import cv2

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
        for cell in row:
            if cell == '1':
                fen_row += '0'
            else:
                fen_row += cell
        fen_rows.append(fen_row)

    return '/'.join(fen_rows)
