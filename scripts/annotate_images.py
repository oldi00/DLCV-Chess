"""
Annotate raw images of chess positions with associated FEN strings.

Assumptions:
- Filename of the FEN strings describes the game name.
- Filename of the FEN strings matches the directory name of the raw images.
- Raw images are able to be sorted via their filenames.
- Every board position has the same amount of images which is specified below.

It is recommended to create the folders `chess_games` and `FENs` within the data
directory since they will already be ignored by git.
"""

import shutil
from pathlib import Path
from tqdm import tqdm

# CONFIGURATION
IMAGES_PER_MOVE = 10
DIR_FEN = Path("data/FENs/")
DIR_IMAGES_RAW = Path("data/chess_games/raw/")
DIR_IMAGES_ANNOTATED = Path("data/chess_games/annotated/")


def process_game(fen_path: Path):
    """Annotate a single game."""

    game_name = fen_path.stem

    with open(fen_path) as f:
        fens = f.readlines()

    # We use underscores to avoid path issues and keep only the first part of the
    # FEN since only it describes the board state.
    fens = [fen.strip().split(" ")[0].replace("/", "_") for fen in fens]

    dir_raw_images = DIR_IMAGES_RAW / game_name
    raw_images = sorted([img for img in dir_raw_images.iterdir()])

    dir_fen_images = DIR_IMAGES_ANNOTATED / game_name
    dir_fen_images.mkdir(parents=True, exist_ok=True)

    if len(fens) * IMAGES_PER_MOVE != len(raw_images):
        print(
            f"[warn] Expected {len(fens) * IMAGES_PER_MOVE} images",
            f"but found {len(raw_images)}. Skipping game '{game_name}'.")
        return

    img_count = 0
    fen_index = 0

    for raw_img_path in tqdm(raw_images, desc=game_name, unit="img"):

        # Create the filename based on the current position (FEN) and add an
        # index to avoid overriding files with the same position.
        fen_img_name = f"{fens[fen_index]}_{img_count}{raw_img_path.suffix}"
        fen_img_path = dir_fen_images / fen_img_name

        shutil.copy2(raw_img_path, fen_img_path)

        # Make sure to update the current position (FEN) once all images of a
        # single position have been annotated.
        img_count += 1
        if img_count == IMAGES_PER_MOVE:
            fen_index += 1
            img_count = 0


def main():

    print(f"[info] Found {len(list(DIR_FEN.iterdir()))} games to annotate.")

    for fen_path in DIR_FEN.iterdir():
        process_game(fen_path)


if __name__ == "__main__":
    main()
