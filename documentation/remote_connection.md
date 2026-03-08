Cluster Setup and Execution Guide

1. Set Up Remote Connection in VS Code
 - To edit code directly on the cluster and use the integrated terminal:
 - Install the Remote - SSH extension in VS Code.
 - Press Ctrl+Shift+P and select Remote-SSH: Connect to Host....
 - Use the command: ssh <your_user>@goethe.hhlr-gu.de.
 - Enter your cluster password when prompted.


2. Set up Home Folder
 - Hint: You can use visual studio code and create a folder called DLCV_Chess in your home /home/<group>/<your_user>/ dir.
 - Otherwise use this in the terminal:
```
mkdir -p /home/<group>/<your_user>/DLCV_Chess/
```

3. Import Code to Home Directory
 - Path: /home/<group>/<your_user>/DLCV_Chess/
 - Open this folder in VS Code via File > Open Folder.
 - You can drag and drop files from your local machine directly into the VS Code explorer to upload them.


4. Setup Everything in the Scratch Folder
 - Path: /scratch/<group>/<your_user>/
 - Create your data and model directories:
```
mkdir -p /scratch/<group>/<your_user>/ChessReD_Hough
mkdir -p /scratch/<group>/<your_user>/chessred/models
```

5. Import files.
 - Path: /scratch/<group>/<your_user>/ChessReD_Hough
 - Import preprocessed images here.

 - Path: /scratch/<group>/<your_user>/chessred
 - Import annotations.json here.
 - Import chessred_hough.pkl, chessred_hough_val.pkl and chessred_hough_test.pkl here.


6. Install Miniconda
Miniconda should be installed in your Home directory.
```
cd /home/<group>/<your_user>
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

7. Install the Environment
Install the actual environment on Scratch to avoid filling up your Home quota.

Create Env:
```
cd /scratch/vihps/<your_user>/
conda create --prefix ./env python=3.10
conda activate ./env
Install ROCm PyTorch (Mandatory for AMD GPUs):
```

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

Install Project Dependencies, currently (11.01.26) those are:
```
pip install numpy matplotlib scikit-learn optuna pydrive2 opencv-python
```

6. Run the submit_job.sh File
```
cd /home/<group>/<your_user>/DLCV_Chess
sbatch submit_job.sh
```

7. Monitor the Job
Check Status: Use squeue or squeue -u <your_user> to monitor your job.


## Folder Structure
/
├── home/<group>/<your_user>/
│   ├── miniconda3/                 <-- Conda Installation
│   └── DLCV_Chess/                 <-- SOURCE CODE
│       ├── config.json
│       ├── submit_job.sh
│       ├── src/
│       │   ├── train_and_eval.py
│       │   └── utils/
│       └── output/                 <-- SLURM LOGS (%j.out / %j.err)
│
└── scratch/<group>/<your_user>/
    ├── env/                        <-- PYTHON ENVIRONMENT
    ├── ChessReD_Hough/             <-- (PREPROCESSED) DATASET
    └── chessred/
        ├── annotations.json
        ├── chessred_hough.pkl      <-- PICKLES
        ├── chessred_hough_val.pkl
        └── chessred_hough_test.pkl
        └── models/                 <-- SAVED TRAINING CHECKPOINTS