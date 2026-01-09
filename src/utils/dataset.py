import torchvision.transforms as transforms
import torch
import pickle

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from utils.preprocess import rotate_fen_90, rotate_fen_180, rotate_fen_270, fen_to_label_vector


# IMPORTANT
# uppercase is white, lowercase is black
class ChessboardRotDataset(Dataset):
    def __init__(self, image_paths, fen_labels, transform=None):
        self.image_paths = image_paths
        self.fen_labels = fen_labels
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
      img_path = self.image_paths[idx]
      fen = self.fen_labels[idx]

      image = Image.open(img_path).convert('RGB')
      image_tensor = self.transform(image)

      try:
          rotations = [
              fen_to_label_vector(fen),
              fen_to_label_vector(rotate_fen_90(fen)),
              fen_to_label_vector(rotate_fen_180(fen)),
              fen_to_label_vector(rotate_fen_270(fen)),
          ]
      except AssertionError as e:
          print(f"Bad FEN at index {idx}:\n{fen}")
          raise e

      return image_tensor, rotations


class ChessboardRotDatasetTEST(Dataset):
    def __init__(self, image_paths, fen_labels, transform=None):
        self.image_paths = image_paths
        self.fen_labels = fen_labels
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
      img_path = self.image_paths[idx]
      fen = self.fen_labels[idx]

      image = Image.open(img_path).convert('RGB')
      image_tensor = self.transform(image)

      try:
          rotations = [
              fen_to_label_vector(fen),
              fen_to_label_vector(rotate_fen_90(fen)),
              fen_to_label_vector(rotate_fen_180(fen)),
              fen_to_label_vector(rotate_fen_270(fen)),
          ]
      except AssertionError as e:
          print(f"Bad FEN at index {idx}:\n{fen}")
          raise e
      image_path = self.image_paths[idx]

      return image_tensor, rotations, image_path
    


### Train Loader
with open("G:/Meine Ablage/DLCV/chessred_hough.pkl", 'rb') as f:
    X_loaded, y_loaded = pickle.load(f)
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(256, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])
def custom_collate(batch):
    images, label_sets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, label_sets  # label_sets is a list of B lists of 4 tensors

# Recreate Dataset and DataLoader
train_dataset = ChessboardRotDataset(X_loaded, y_loaded, transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)

print(f"Total images: {len(train_dataset)}")



### Val Loader
with open("G:/Meine Ablage/DLCV/chessred_hough.pkl", 'rb') as f:
    X_loaded, y_loaded = pickle.load(f)
print(X_loaded)
print(y_loaded)
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Recreate Dataset and DataLoader
val_dataset = ChessboardRotDataset(X_loaded, y_loaded, transform=data_transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
print(f"Total images: {len(val_dataset)}")



### Test Loader
with open("G:/Meine Ablage/DLCV/chessred_hough.pkl", 'rb') as f:
    X_loaded, y_loaded = pickle.load(f)
print(X_loaded)
print(y_loaded)
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
def custom_collateTEST(batch):
    images, label_sets, paths = zip(*batch)  # Now unpack 3 values
    images = torch.stack(images, dim=0)
    return images, label_sets, paths
def custom_collate(batch):
    # Unpack only the first two values: image_tensor and rotations
    images, label_sets, _ = zip(*batch)  # Ignore paths
    images = torch.stack(images, dim=0)
    return images, label_sets


# Recreate Dataset and DataLoader
test_dataset = ChessboardRotDatasetTEST(X_loaded, y_loaded, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=custom_collateTEST)
print(f"Total images: {len(test_dataset)}")

test_dataset_plain = ChessboardRotDataset(X_loaded, y_loaded, transform=data_transform)
test_loader_plain = DataLoader(test_dataset_plain, batch_size=16, shuffle=True, collate_fn=custom_collate)
print(f"Total images: {len(test_dataset_plain)}")
