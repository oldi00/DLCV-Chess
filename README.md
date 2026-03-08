# DLCV-Chess

## Authors
- Luis Göppel
- Markus Oldenburger
- Miles Lenz

# Board Detection and Dataset Pipeline

This module handles the extraction of top-down chessboards from raw photographs, the generation of synthetic and real-world training datasets, and the evaluation of the detection algorithms. 

## Architecture & Design

* **Single Source of Code**: A single core detector (`board_detection.py`) is used across the entire project, from dataset generation to evaluation and inference to guarantee consistent behavior.
* **Separation of Training Stages**: Dataset generation is split into `pretrain` (synthetic data) and `finetune` (real/custom data) scripts. This isolates the differing processing needs, source formats, and trust levels of the data.
* **Evaluation**: The `eval.py` script strictly benchmarks the detector against a classical baseline, allowing for objective measurement of improvements independent of the production pipeline.
* **Modular Utilities**: Reusable computer vision operations and visualization tools are decoupled from the main pipelines to reduce code duplication and facilitate experimentation.

---

## Core Detection

### `script/board_detection.py`
The primary board detector used throughout the project. It processes a raw photo to extract a cropped, top-down view of the chessboard.
* **Pipeline**: Isolates the foreground using `rembg`, applies a binary mask with morphological cleanup, identifies the largest valid contour based on size and solidity, approximates a quadrilateral, and performs a perspective warp.
* **Output**: Calculates a confidence score based on contour solidity and quadrilateral fit quality. It provides a CLI/`stdin` interface that outputs JSON (`warped_image_bytes`, `confidence_score`, `detected_corners`) or returns `NO_BOARD_DETECTED`.
* **Deployment**: Designed to be packaged as a standalone executable (PyInstaller instructions are included in the file docstring).

---

## Dataset Generation

### `build_pretrain_dataset.py`
Builds a unified synthetic dataset for pretraining by merging data from Unity, Kaggle SynthChess, Kaggle ChessRender360, and Blender.
* Normalizes differing annotation formats (corner coordinates + FEN labels).
* Warps boards to a consistent top-down perspective.
* Utilizes atomic writes for `metadata.json` (via temporary file renaming) to prevent data corruption during interrupted runs.

### `build_finetune_dataset.py`
Generates the fine-tuning dataset using real-world images and the ChessReD dataset.
* Executes the core detector over candidate images utilizing multiprocessing.
* Saves successfully warped boards alongside per-sample metadata (FEN and confidence scores).
* Tracks failed detections (`metadata["fails"]`) to skip repeatedly failing images on subsequent executions.
* Includes a `sync_chessred()` mode to directly synchronize ChessReD samples.

### `annotation.py`
Automates the labeling of raw game images by embedding FEN strings directly into the output filenames.
* Assumes a fixed sequence of images per move (default: `10`) and maps image chunks to specific FEN lines.
* Moves files from `raw/<game>/` to `annotated/<game>/` using sanitized, FEN-based nomenclature.
* Skips games with frame count mismatches (`#FEN * 10 != #images`) to prevent assigning incorrect labels.

---

## Evaluation & Utilities

### `eval.py`
Benchmarks the performance of the board corner detection algorithm.
* Compares the primary pipeline against a classical computer vision baseline.
* Evaluates across Unity, ChessReD, and custom datasets utilizing dataset-specific ground-truth extraction logic.
* Calculates polygon Intersection over Union (IoU) utilizing the `shapely` library.
* Reports average IoU and success rates at IoU > 0.5 and > 0.8, saving the final metrics to `eval.json`.

### `utils_chess_cv.py`
Shared computer vision primitives tailored for chessboards.
* Handles robust corner ordering, perspective warping, and piece-annotation-to-FEN conversions.
* Contains the classical baseline detector (`detect_board_corners`) used for comparison in the evaluation script.

### `visualization.py`
Debugging and plotting tools for intermediate computer vision steps.
* Generates image grids and histogram plots.
* Draws detected board polygons and overlays 8x8 grids on warped boards to visually verify alignment.

### `utils.py`
General utility module for basic operations.
* Includes standard RGB image loading (`load_image_RGB`).
* Provides aspect-ratio-preserving resizing with dynamic interpolation based on upscaling or downscaling requirements.


# Model Training, Evaluation, and Deployment

This section of the repository contains the core deep learning pipeline for training, evaluating, and exporting the chessboard recognition CNN. The codebase is designed to run on a SLURM-based cluster using PyTorch Distributed Data Parallel.

## Core Training Scripts

* **`src/train_model.py`**: The primary pre-training script. It trains the base CNN architecture from scratch using synthetic data.
* **`src/finetune.py`**: Handles transfer learning on real-world datasets.

## Utilities (`src/utils/`)

* **`models.py`**: Defines the PyTorch neural network architectures, primarily `CustomChessCNN_v3`, which utilizes Pre-Activation Residual Blocks and adaptive pooling to output a 64x13 tensor (64 squares, 13 classes).
* **`dataset.py`**: Contains the PyTorch `Dataset` and `DataLoader` implementations.
* **`preprocess.py`**: A computer vision utility for processing raw board images into structured `.pkl` datasets. Handles FEN string expansion/compression, dataset splitting, and image transformations.
* **`inspect_pickle.py` & `update_pickle.py`**: Helper scripts to verify the contents and lengths of generated datasets or safely migrate absolute paths within the pickle files.

## Evaluation (`src/evaluation/`)

* **`eval.py`**: An evaluation suite that calculates metrics. It computes square-level accuracy, board-level match rates, errors per board, Expected Calibration Error for reliability, and per-class precision/recall/F1 scores. 

## Inference & Deployment (`src/app/`)

* **`export_to_onnx.py`**: Converts the trained PyTorch `.pth` models into the ONNX format for lightweight deployment.
* **`chess_predictor.py`**: A standalone inference script utilizing `onnxruntime` to process images and output piece probabilities via a standard input/output interface. Designed to be packaged as a compiled executable.
* **`compare_inference.py`**: A validation script that runs identical images through both the raw PyTorch model and the compiled ONNX.

## Cluster Execution (`bash_scripts/`)

* Contains SLURM batch scripts (`train_model.sh`, `finetune.sh`, `preprocess.sh`, `inspect_pickle.sh`) for submitting to a cluster.