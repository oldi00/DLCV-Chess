import os
import torch
import json
import argparse
from utils.models import CustomChessCNN_v3
from train_model import train
from utils.dataset import get_train_loader, get_val_loader

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def freeze_layers(model, freeze_level):
    """
    Dynamically freezes the network up to the specified layer.
    Hierarchy: stem -> layer1 -> layer2 -> layer3
    """
    print(f"\nFreezing strategy: {freeze_level.upper()}")

    # Define the hierarchical order of your CNN blocks
    hierarchy = ["stem", "layer1", "layer2", "layer3"]

    if freeze_level == "none":
        print("No layers frozen. Full fine-tuning.")
    elif freeze_level not in hierarchy:
        print(
            f"Warning: Unknown freeze level '{freeze_level}'. Valid options are: none, stem, layer1, layer2, layer3."
        )
        print("Proceeding with NO freezing.")
    else:
        # Loop through the hierarchy and freeze until we hit the target level
        for layer_name in hierarchy:
            module = getattr(model, layer_name)
            for param in module.parameters():
                param.requires_grad = False
            print(f"{layer_name} frozen.")

            # Stop freezing once we've frozen the requested level
            if layer_name == freeze_level:
                break

    # Verify and print status
    print("\nLayer Status Summary:")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    print(
        f"Trainable Parameters: {trainable_params:,} / {all_params:,} ({(trainable_params / all_params):.1%})"
    )
    print("-" * 30)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Configuration Base
    # We will overwrite the paths with our arguments.
    config_path = "/home/vihps/vihps01/DLCV_Chess/config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}

    # Initialize Model
    print("Initializing CustomChessCNN_v3...")
    model = CustomChessCNN_v3(num_classes=13, dropout=0.3).to(device)

    # Load Pre-trained Synthetic Weights
    if not os.path.exists(args.weights):
        print(f"Error: Could not find model weights at {args.weights}")
        return

    print(f"Loading synthetic weights from: {args.weights}")
    state_dict = torch.load(args.weights, map_location=device)

    # Handle DDP 'module.' prefix if present
    new_state_dict = {
        k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()
    }
    model.load_state_dict(new_state_dict)
    print("Weights loaded.")

    # Apply Freezing
    freeze_layers(model, args.freeze_level)
    # DataLoaders
    print("\nPreparing DataLoaders for Fine-tuning...")
    print(f"Train Data: {args.train_pkl}")
    print(f"Val Data:   {args.val_pkl}")

    # Inject the command-line arguments directly into the config dictionary
    config["train_pickle_path"] = args.train_pkl
    config["cluster_train_pickle_path"] = args.train_pkl
    config["val_pickle_path"] = args.val_pkl
    config["cluster_val_pickle_path"] = args.val_pkl

    train_loader = get_train_loader(config, batch_size=args.batch_size, use_ddp=False)
    val_loader = get_val_loader(config, batch_size=args.batch_size, use_ddp=False)

    # Start Training
    print("\nStarting Fine-tuning")
    print(f"Target Save Directory: {args.save_dir}")

    train(
        model=model,
        dataloader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_model=True,
        save_dir=args.save_dir,
        val_loader=val_loader,
        rank=0,
        use_ddp=False,
        patience=args.patience,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the chess model")

    # Required Paths
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to the pre-trained .pth model"
    )
    parser.add_argument(
        "--train_pkl", type=str, required=True, help="Path to the real train data .pkl"
    )
    parser.add_argument(
        "--val_pkl", type=str, required=True, help="Path to the real val data .pkl"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the fine-tuned checkpoints",
    )

    # Hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience"
    )

    # Freezing Argument
    parser.add_argument(
        "--freeze_level",
        type=str,
        default="layer1",
        choices=["none", "stem", "layer1", "layer2", "layer3"],
        help="Freeze the network up to this level. Options: none, stem, layer1, layer2, layer3",
    )

    args = parser.parse_args()
    main(args)
