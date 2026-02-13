import os
import torch
import json
import argparse
from pathlib import Path

# Imports from your source tree
from models import CustomChessCNN_v3
from utils.dataset import get_test_loader, get_val_loader, get_train_loader
from evaluation.engine import evaluate_comprehensive


def main():
    parser = argparse.ArgumentParser(description="Run Comprehensive Chess Evaluation")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config.json"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to .pth model weights. Overrides config if provided.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    print(f"Loading {args.split} data...")
    if args.split == "test":
        loader = get_test_loader(config, batch_size=args.batch_size)
    elif args.split == "val":
        loader = get_val_loader(config, batch_size=args.batch_size)
    else:
        loader = get_train_loader(config, batch_size=args.batch_size)

    # 3. Load Model
    print("Initializing model...")
    model = CustomChessCNN_v3(
        num_classes=13, dropout=0.0
    )  # Dropout 0 for inference usually

    # Determine weights path
    weights_path = args.weights if args.weights else config.get("best_model_weights")
    if not weights_path or not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        return

    print(f"Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)

    # Handle DataParallel wrapping ('module.' prefix)
    state_dict = checkpoint.get(
        "state_dict", checkpoint.get("model_state_dict", checkpoint)
    )
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)

    # 4. Run Evaluation
    results = evaluate_comprehensive(
        model=model,
        dataloader=loader,
        device=device,
        plot=(not args.no_plot),
        title_prefix=f"Eval_{args.split.upper()}",
    )

    # Optional: Save results dict
    # np.save(f"eval_results_{args.split}.npy", results)


if __name__ == "__main__":
    main()
