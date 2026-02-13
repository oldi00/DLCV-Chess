import pickle


# --- Configuration ---
input_pkl = "G:/Meine Ablage/DLCV/models/chessred/pkl/cluster_chessred_hough_test.pkl"
output_pkl = "G:/Meine Ablage/DLCV/models/chessred/pkl/chessred_hough_test.pkl"

# Example: Cluster was /scratch/vihps/vihps01/real_images, Drive is G:/Meine Ablage/DLCV/images.
old_path_root = "/scratch/vihps/vihps01/legacy/ChessReD_Hough"
new_path_root = "G:/Meine Ablage/DLCV/ChessReD_Hough"


def update_paths(data, old_prefix, new_prefix):
    """
    Recursively traverse data structures to find and replace path strings.
    """
    if isinstance(data, str):
        # Check if the string starts with the old prefix
        if data.startswith(old_prefix):
            return data.replace(old_prefix, new_prefix)
        return data

    elif isinstance(data, dict):
        return {k: update_paths(v, old_prefix, new_prefix) for k, v in data.items()}

    elif isinstance(data, list):
        return [update_paths(item, old_prefix, new_prefix) for item in data]

    elif isinstance(data, tuple):
        # Tuples are immutable, so we convert to list, update, and convert back
        return tuple(update_paths(item, old_prefix, new_prefix) for item in data)

    # Return data unchanged if it's not a container or string (e.g., int, float)
    return data


# --- Execution ---
try:
    with open(input_pkl, "rb") as f:
        content = pickle.load(f)

    # specific handling for pandas DataFrames (see Option 2)
    # otherwise, use the recursive function:
    new_content = update_paths(content, old_path_root, new_path_root)

    with open(output_pkl, "wb") as f:
        pickle.dump(new_content, f)

    print(f"Success! Re-wrote paths and saved to {output_pkl}")

except FileNotFoundError:
    print(f"Error: Could not find file {input_pkl}")
except Exception as e:
    print(f"An error occurred: {e}")
