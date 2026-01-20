import torch
import torch.onnx
from models import CustomChessCNN_v3
import os

MODEL_PATH = "epoch15.pth"
OUTPUT_ONNX = "chess_model.onnx"


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

    print(f"Exporting to {OUTPUT_ONNX}...")
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
    print("✅ Export complete!")


if __name__ == "__main__":
    export_model()
