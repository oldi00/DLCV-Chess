import os
import torch
import pickle
import multiprocessing
import torchvision.transforms as transforms

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from utils.preprocess import (
    rotate_fen_90,
    rotate_fen_180,
    rotate_fen_270,
    fen_to_label_vector,
)


def get_path_from_config_file(config, key):
    """
    Checks for SLURM environment variables to determine if running on cluster.
    If on cluster, looks for 'cluster_' prefix in config.
    """
    on_cluster = "SLURM_JOB_ID" in os.environ
    if on_cluster:
        cluster_key = f"cluster_{key}"
        print(f"Cluster uses the following key: {cluster_key}")
        return config.get(cluster_key, config.get(key))
    return config.get(key)


# Helper to get worker count
def get_optimal_workers():
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    try:
        return min(8, multiprocessing.cpu_count())
    except:
        return 4


# IMPORTANT
# uppercase is white, lowercase is black
class ChessboardRotDataset(Dataset):
    def __init__(self, image_paths, fen_labels, transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.ToTensor()
        
        # Pre-compute Rotations. Compute once at the start
        print("Pre-computing FEN rotations")
        self.cached_rotations = []
        for i, fen in enumerate(fen_labels):
            try:
                rotations = [
                    fen_to_label_vector(fen),
                    fen_to_label_vector(rotate_fen_90(fen)),
                    fen_to_label_vector(rotate_fen_180(fen)),
                    fen_to_label_vector(rotate_fen_270(fen)),
                ]
                self.cached_rotations.append(rotations)
            except AssertionError as e:
                print(f"Bad FEN at index {i}:\n{fen}")
                raise e
        print("Done pre-computing.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)        
        rotations = self.cached_rotations[idx]

        return image_tensor, rotations


class ChessboardRotDatasetTEST(Dataset):
    def __init__(self, image_paths, fen_labels, transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.ToTensor()
        
        # Efficiency
        print("Pre-computing FEN rotations (TEST)")
        self.cached_rotations = []
        for i, fen in enumerate(fen_labels):
            try:
                rotations = [
                    fen_to_label_vector(fen),
                    fen_to_label_vector(rotate_fen_90(fen)),
                    fen_to_label_vector(rotate_fen_180(fen)),
                    fen_to_label_vector(rotate_fen_270(fen)),
                ]
                self.cached_rotations.append(rotations)
            except AssertionError:
                # Fallback or skip
                pass 
        print("Done.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        rotations = self.cached_rotations[idx]
        
        return image_tensor, rotations, img_path


def custom_collate(batch):
    # Unpack only the first two values: image_tensor and rotations
    images, label_sets = zip(*batch)  # Ignore paths
    images = torch.stack(images, dim=0)
    return images, label_sets


def custom_collateTEST(batch):
    images, label_sets, paths = zip(*batch)  # Now unpack 3 values
    images = torch.stack(images, dim=0)
    return images, label_sets, paths


def get_train_loader(config, batch_size=16, use_ddp=False):
    ### Train Loader
    pkl_path = get_path_from_config_file(config, "train_pickle_path")

    with open(pkl_path, "rb") as f:
        X_loaded, y_loaded = pickle.load(f)

    data_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomRotation(5),
            transforms.RandomResizedCrop(256, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ChessboardRotDataset(X_loaded, y_loaded, transform=data_transform)
    
    # DDP
    sampler = None
    if use_ddp:
        sampler = DistributedSampler(train_dataset, shuffle=True)

    # Worker & Pin Memory for efficiency
    num_workers = get_optimal_workers()
    print(f"Using {num_workers} workers for Train Loader")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Total images train loader {len(train_dataset)}")

    return train_loader


def get_val_loader(config, batch_size=16, use_ddp=False):
    ### Val Loader
    pkl_path = get_path_from_config_file(config, "val_pickle_path")

    with open(pkl_path, "rb") as f:
        X_loaded, y_loaded = pickle.load(f)

    data_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    val_dataset = ChessboardRotDataset(X_loaded, y_loaded, transform=data_transform)
    
    # DDP
    sampler = None
    if use_ddp:
        sampler = DistributedSampler(val_dataset, shuffle=False)

    num_workers = get_optimal_workers()
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=sampler,
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Total images val loader: {len(val_dataset)}")

    return val_loader


def get_test_loader(config, batch_size=16):
    ### Test Loader
    pkl_path = get_path_from_config_file(config, "test_pickle_path")

    with open(pkl_path, "rb") as f:
        X_loaded, y_loaded = pickle.load(f)

    data_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    test_dataset = ChessboardRotDatasetTEST(
        X_loaded, y_loaded, transform=data_transform
    )
    
    num_workers = get_optimal_workers()

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collateTEST,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    print(f"Total images: {len(test_dataset)}")

    return test_loader