import os
import torch
import json
import argparse
from models import CustomChessCNN_v3
from train_and_eval import evaluate_plus, get_path_from_config_file
from utils.dataset import get_test_loader

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default Config Path
CONFIG_FILENAME = "config.json"
DEFAULT_CONFIG_PATH = "/home/vihps/vihps01/DLCV_Chess/config.json"

# PATH TO REAL TEST DATA 
REAL_TEST_PICKLE_PATH = "/scratch/vihps/vihps01/legacy/chessred/chessred_hough_test.pkl"

def main(model_path):
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Configuration 
    config_path = CONFIG_FILENAME if os.path.exists(CONFIG_FILENAME) else DEFAULT_CONFIG_PATH
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Config loaded from {config_path}")
    else:
        print(f"Error: Config file not found at {config_path}")
        return

    # Initialize Model
    print("Initializing the model")
    model = CustomChessCNN_v3(num_classes=13, dropout=0.3).to(device)

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
    
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle DDP module prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded and set to eval mode.")

    # Prepare Test Loader
    print(f"Loading Test Data from: {REAL_TEST_PICKLE_PATH}")
    
    if not os.path.exists(REAL_TEST_PICKLE_PATH):
        print(f"Error: Test pickle not found at {REAL_TEST_PICKLE_PATH}")
        return

    # Override Config in Memory
    config["test_pickle_path"] = REAL_TEST_PICKLE_PATH
    config["cluster_test_pickle_path"] = REAL_TEST_PICKLE_PATH
    
    # Get Loader
    test_loader = get_test_loader(config, batch_size=16)

    # Run Evaluation
    print("\nStarting Evaluation on REAL TEST set.")
    metrics = evaluate_plus(
        model=model,
        dataloader=test_loader,
        device=device,
        compute_plots=False,
        title_prefix="Real Test Set"
    )

    # --- 7. Print Summary ---
    print("\n" + "="*40)
    print("       FINAL TEST RESULTS       ")
    print("="*40)
    print(f"Square Accuracy (All):       {metrics['square_acc_all']:.2%}")
    print(f"Square Accuracy (Occupied):  {metrics['square_acc_non_empty']:.2%}")
    print(f"Board Accuracy (Exact):      {metrics['exact_board_acc']:.2%}")
    print(f"Mean Errors per Board:       {metrics['board_error_mean']:.2f}")
    print("-" * 40)
    print("Board Error Histogram:")
    for k, v in metrics['board_error_hist'].items():
        print(f"  {k} errors: {v} boards")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the .pth model file to test")
    args = parser.parse_args()
    
    main(args.model_path)