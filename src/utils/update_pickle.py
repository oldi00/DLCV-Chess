import pickle


# --- Configuration ---
input_pkl = (
    "G:/Meine Ablage/DLCV/models/finetuned (new)/pkl/new_cluster_real_test_data.pkl"
)
output_pkl = "G:/Meine Ablage/DLCV/models/finetuned (new)/pkl/new_real_test_data.pkl"

# Example: Cluster was /scratch/vihps/vihps01/real_images, Drive is G:/Meine Ablage/DLCV/images.
old_path_root = "/scratch/vihps/vihps01/real_images"
new_path_root = "G:/Meine Ablage/DLCV/Fine-Tuning Dataset"


def update_paths(data, old_prefix, new_prefix):
    if isinstance(data, str):
        if data.startswith(old_prefix):
            return data.replace(old_prefix, new_prefix)
        return data

    elif isinstance(data, dict):
        return {k: update_paths(v, old_prefix, new_prefix) for k, v in data.items()}

    elif isinstance(data, list):
        return [update_paths(item, old_prefix, new_prefix) for item in data]

    elif isinstance(data, tuple):
        return tuple(update_paths(item, old_prefix, new_prefix) for item in data)
    return data


try:
    with open(input_pkl, "rb") as f:
        content = pickle.load(f)
    new_content = update_paths(content, old_path_root, new_path_root)

    with open(output_pkl, "wb") as f:
        pickle.dump(new_content, f)

    print(f"Success! Rewrote paths and saved to {output_pkl}")

except FileNotFoundError:
    print(f"Error: Could not find file {input_pkl}")
except Exception as e:
    print(f"An error occurred: {e}")
