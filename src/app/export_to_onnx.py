import torch
import torch.onnx
import os
import sys

# Get the directory where this current script is located
# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.models import CustomChessCNN_v3


MODEL_PATH = r"G:/Meine Ablage/DLCV/models/finetuned (new)/epoch43_layer1_freeze_model/best_model.pth"
OUTPUT_ONNX = r"C:/Users/olden/Desktop/programming/DLCV-Chess/onnx_models/finetuned_chess_model_epoch43.onnx"


def export_model():
    device = torch.device("cpu")

    model = CustomChessCNN_v3()
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)

    print(f"Exporting to {OUTPUT_ONNX}")
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_ONNX,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("Export complete")


if __name__ == "__main__":
    export_model()
