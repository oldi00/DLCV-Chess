import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import os

MODELS_DIR = "onnx_models"
IMAGE_DIR = "G:/Meine Ablage/DLCV/Fine-Tuning"

IDX_TO_PIECE = {
    0: "P",
    1: "R",
    2: "N",
    3: "B",
    4: "Q",
    5: "K",
    6: "p",
    7: "r",
    8: "n",
    9: "b",
    10: "q",
    11: "k",
    12: ".",
}


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def load_all_models(models_dir):
    """Lädt alle .onnx Modelle in ein Dictionary."""
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".onnx")]
    loaded_models = {}

    for m_file in model_files:
        path = os.path.join(models_dir, m_file)
        try:
            session = ort.InferenceSession(path)
            loaded_models[m_file] = session
        except Exception as e:
            print(f"Fehler bei {m_file}: {e}")

    return loaded_models


def draw_prediction_on_axis(ax, img_rgb, probs, model_name):
    # VON GEMINI ERSTELLT
    display_img = cv2.resize(img_rgb, (400, 400))
    ax.imshow(display_img)
    max_probs = np.max(probs, axis=1)
    avg_conf = np.mean(max_probs)

    ax.set_title(f"{model_name}\nØ Sicherheit: {avg_conf * 100:.1f}%", fontsize=10)
    ax.axis("off")

    step = 400 / 8
    for i in range(1, 8):
        ax.axhline(i * step, color="white", linestyle="-", alpha=0.15)
        ax.axvline(i * step, color="white", linestyle="-", alpha=0.15)

    for i, p in enumerate(probs):
        row, col = divmod(i, 8)
        best_idx = np.argmax(p)
        text = IDX_TO_PIECE[best_idx]

        if best_idx != 12:
            x = col * step + (step / 2)
            y = row * step + (step / 2)

            color = "yellow" if best_idx <= 5 else "cyan"

            # Falls unsicher (<80%), roten Rand um den Text machen
            bbox_props = dict(facecolor="black", alpha=0.6, edgecolor="none")
            if p[best_idx] < 0.8:
                bbox_props["edgecolor"] = "red"
                bbox_props["linewidth"] = 2

            ax.text(
                x,
                y,
                text,
                color=color,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                bbox=bbox_props,
            )


def main():
    models = load_all_models(MODELS_DIR)
    valid_ext = (".png", ".jpg", ".jpeg")
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_ext)]

    for img_name in image_files:
        full_path = os.path.join(IMAGE_DIR, img_name)
        img_bgr = cv2.imread(full_path)

        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img_rgb, (256, 256))
        img_tensor = img_resized.astype(np.float32) / 255.0
        img_tensor = img_tensor.transpose(2, 0, 1)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        n_cols = 1 + len(models)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
        if n_cols == 1:
            axes = [axes]
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Original: {img_name}")
        axes[0].axis("off")

        print(f"\n--- Analyse: {img_name} ---")

        for i, (model_name, session) in enumerate(models.items()):
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: img_tensor})
            probs = softmax(outputs[0][0])
            draw_prediction_on_axis(axes[i + 1], img_rgb, probs, model_name)

            print(f"Modell '{model_name}': fertig.")

        plt.tight_layout()
        plt.show()  # Fenster schließen, um zum nächsten Bild zu kommen


if __name__ == "__main__":
    main()
