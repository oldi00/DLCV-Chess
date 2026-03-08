import pickle
import argparse
from pathlib import Path


def inspect_pickle(pkl_path, num_samples=5):
    path = Path(pkl_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"\n{'=' * 50}")
    print(f"Inspecting: {path.name}")
    print(f"{'=' * 50}")

    # Load the pickle file
    with open(path, "rb") as f:
        X, y = pickle.load(f)

    total_samples = len(X)
    print(f"Total samples: {total_samples}")
    if total_samples == 0:
        print("Dataset is empty!")
        return

    print(f"Data type of X: {type(X)}, Elements: {type(X[0])}")
    print(f"Data type of y: {type(y)}, Elements: {type(y[0])}\n")

    # Show a few examples
    samples_to_show = min(total_samples, num_samples)
    print(f"Showing first {samples_to_show} pairs:\n")

    for i in range(samples_to_show):
        print(f"Sample {i + 1}:")
        print(f"Image Path (X): {X[i]}")
        print(f"FEN String (y): {y[i]}")

        # An expanded FEN string should be exactly 71 chars (64 pieces/blanks + 7 slashes)
        fen_length = len(y[i])
        status = "Correct" if fen_length == 71 else "Incorrect"
        print(f"FEN Length: {fen_length} chars ({status})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect generated Pickle dataset files."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Directory where .pkl files are saved",
    )
    parser.add_argument("--prefix", type=str, required=True, help="Output prefix used")
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of samples to print per split"
    )

    args = parser.parse_args()
    base_path = Path(args.base_dir)

    # Automatically loop through the possible splits
    splits = ["train", "val", "test"]

    for split in splits:
        pkl_file = base_path / f"{args.prefix}_{split}_data.pkl"
        inspect_pickle(pkl_file, num_samples=args.samples)
