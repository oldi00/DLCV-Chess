import json
import os

def generate_configs():
    # Folder to store your 8 unique parameters
    config_dir = "parameters"
    os.makedirs(config_dir, exist_ok=True)
    
    # Example: Testing different learning rates across the 8 GPUs
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-4, 5e-5, 1e-5, 5e-6]
    dropouts = [0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5]

    for i in range(8):
        config = {
            "task_id": i,
            "lr": learning_rates[i],
            "dropout": dropouts[i],
            "epochs": 15,
            "batch_size": 16
        }
        
        with open(f"{config_dir}/{i}.params", "w") as f:
            json.dump(config, f, indent=4)
    
    print(f"✅ Generated 8 config files in {config_dir}/")

if __name__ == "__main__":
    generate_configs()