import os
import torch
import json
import torch.nn as nn
from models import CustomChessCNN_v3
from train_and_eval import (
    train,
    get_train_loader,
    get_val_loader,
    get_path_from_config_file,
)

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# File Paths
CONFIG_FILENAME = "config.json"
DEFAULT_CONFIG_PATH = "/home/vihps/vihps01/DLCV_Chess/config.json"
PRETRAINED_WEIGHTS_PATH = "/scratch/vihps/vihps01/unity/models/epoch35.pth"

# Fine-tuning Hyperparameters
FINETUNE_LEARNING_RATE = 5e-5
FINETUNE_EPOCHS = 15

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def freeze_geometric_layers(model):
    """
    Freezes the early layers (Stem + Layer1) which are responsible for
    geometric features (lines, corners) that are stable between Sim and Real.
    """
    print("\n🔒 Freezing Geometric Layers...")

    # 1. Freeze Stem (Initial 7x7 Conv)
    for param in model.stem.parameters():
        param.requires_grad = False
    print("   -> Stem frozen.")

    # 2. Freeze Layer 1
    for param in model.layer1.parameters():
        param.requires_grad = False
    print("   -> Layer 1 frozen.")

    # Verify Status
    print("\n🔍 Layer Status:")
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"   -> Trainable Parameters: {trainable_params:,} / {all_params:,} ({(trainable_params/all_params):.1%})"
    )
    print("-" * 30)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    # --- 1. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # --- 2. Load Configuration ---
    # Try local config first, then fallback to cluster default
    config_path = (
        CONFIG_FILENAME if os.path.exists(CONFIG_FILENAME) else DEFAULT_CONFIG_PATH
    )

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"📂 Config loaded from {config_path}")
    else:
        print(f"❌ Error: Config file not found at {config_path}")
        return

    # --- 3. Initialize Model ---
    print("🏗️  Initializing CustomChessCNN_v3...")
    model = CustomChessCNN_v3(num_classes=13, dropout=0.3).to(device)

    # --- 4. Load Pre-trained Synthetic Weights ---
    if not os.path.exists(PRETRAINED_WEIGHTS_PATH):
        print(f"❌ Error: {PRETRAINED_WEIGHTS_PATH} not found. Please check the path.")
        return

    print(f"📥 Loading synthetic weights from {PRETRAINED_WEIGHTS_PATH}...")
    state_dict = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=device)

    # Handle DDP 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    print("✅ Weights loaded.")

    # --- 5. Apply Sim2Real Freezing Strategy ---
    freeze_geometric_layers(model)

    # --- 6. Prepare DataLoaders (Swapping Paths) ---
    print("📦 preparing DataLoaders for Fine-tuning...")

    # HACK: We overwrite the 'train_pickle_path' in the config dictionary in-memory.
    # This tricks 'get_train_loader' into loading the REAL data (finetune path)
    # instead of the SYNTHETIC data, without changing the library code.

    real_train_path = get_path_from_config_file(config, "finetune_train_pickle_path")
    real_val_path = get_path_from_config_file(config, "finetune_val_pickle_path")

    # Overwrite config in memory
    config["train_pickle_path"] = real_train_path
    config["val_pickle_path"] = real_val_path

    print(f"   -> Training Data:   {real_train_path}")
    print(f"   -> Validation Data: {real_val_path}")

    train_loader = get_train_loader(config, batch_size=16, use_ddp=False)
    val_loader = get_val_loader(config, batch_size=16, use_ddp=False)

    # --- 7. Determine Save Directory ---
    # Use specific finetune save dir if available, else subdirectory of model_save_dir
    ft_save_dir = get_path_from_config_file(config, "finetune_save_dir")
    if not ft_save_dir:
        base_save_dir = get_path_from_config_file(config, "model_save_dir")
        ft_save_dir = os.path.join(base_save_dir, "finetuned_real")

    print(f"\n🏃 Starting Fine-tuning (Sim2Real)...")
    print(f"   Target Save Directory: {ft_save_dir}")

    # --- 8. Start Training ---
    train(
        model=model,
        dataloader=train_loader,
        device=device,
        epochs=FINETUNE_EPOCHS,
        lr=FINETUNE_LEARNING_RATE,
        save_model=True,
        save_dir=ft_save_dir,
        val_loader=val_loader,
        rank=0,
        use_ddp=False,
    )


if __name__ == "__main__":
    main()
