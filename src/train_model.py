import os
import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributed as dist

from evaluation.eval import evaluate_rotation_invariant
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.models import CustomChessCNN_v3
from utils.dataset import (
    get_train_loader,
    get_val_loader,
)
from tqdm import tqdm


TRAINING_PLOT_FILENAME = "training_plot.png"
CHECKPOINT_PREFIX = "epoch"
CONFIG_MODEL_SAVE_KEY = "model_save_dir"


def train(
    model,
    dataloader,
    device,
    epochs=35,
    save_model=True,
    save_dir=None,
    plot_loss=True,
    lr=1e-4,
    weight_decay=1e-5,
    empty_class_idx=12,
    empty_class_weight=0.3,
    val_loader=None,
    rank=0,
    use_ddp=False,
    patience=None,
):
    if not use_ddp:
        model = model.to(device)

    weights = torch.ones(13)
    weights[empty_class_idx] = empty_class_weight
    weights = weights.to(device)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    epoch_train_losses = []
    epoch_train_acc_all = []
    epoch_train_acc_no_empty = []

    epoch_val_losses = []
    epoch_val_acc_all = []
    epoch_val_acc_no_empty = []

    if save_model and rank == 0:
        assert save_dir is not None, (
            "Please provide a save_dir to save model checkpoints."
        )
        os.makedirs(save_dir, exist_ok=True)

    # Early Stopping Trackers
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        if use_ddp:
            dataloader.sampler.set_epoch(epoch)

        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                # If the BN layer has weights and they are frozen, lock it to eval
                if module.weight is not None and not module.weight.requires_grad:
                    module.eval()

        running_loss = 0.0
        correct_all = 0
        correct_non_empty = 0
        total_all = 0
        total_non_empty = 0

        if rank == 0:
            iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        else:
            iterator = dataloader

        for images, label_rotations in iterator:
            images = images.to(device)
            label_rotations = [
                [lbl.to(device) for lbl in rotations] for rotations in label_rotations
            ]

            optimizer.zero_grad()
            outputs = model(images)  # (B, 64, 13)

            losses = []
            matched_labels = []
            for i in range(images.size(0)):
                sample_losses = [
                    F.cross_entropy(outputs[i], lbl, weight=weights, ignore_index=-1)
                    for lbl in label_rotations[i]
                ]
                best_idx = torch.argmin(torch.stack(sample_losses))
                losses.append(sample_losses[best_idx])
                matched_labels.append(label_rotations[i][best_idx])
            loss = torch.stack(losses).mean()
            matched_labels = torch.stack(matched_labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = outputs.argmax(dim=2)
            mask = matched_labels != empty_class_idx
            correct_all += (preds == matched_labels).sum().item()
            total_all += matched_labels.numel()
            correct_non_empty += ((preds == matched_labels) & mask).sum().item()
            total_non_empty += mask.sum().item()

        avg_train_loss = running_loss / len(dataloader)
        acc_all = correct_all / total_all if total_all > 0 else 0.0
        acc_no_empty = (
            correct_non_empty / total_non_empty if total_non_empty > 0 else 0.0
        )

        epoch_train_losses.append(avg_train_loss)
        epoch_train_acc_all.append(acc_all)
        epoch_train_acc_no_empty.append(acc_no_empty)

        if rank == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_train_loss:.4f} | Acc (all): {acc_all:.4f} | Acc (non-empty): {acc_no_empty:.4f}"
            )

        if save_model and rank == 0:
            save_path = os.path.join(save_dir, f"epoch{epoch + 1}.pth")
            state_dict = model.module.state_dict() if use_ddp else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"Model saved to: {save_path}")

        # --- Evaluate ---
        if val_loader is not None:
            val_loss, val_acc_all, val_acc_no_empty, _ = evaluate_rotation_invariant(
                model, val_loader, device=device, rank=rank
            )
            epoch_val_losses.append(val_loss)
            epoch_val_acc_all.append(val_acc_all)
            epoch_val_acc_no_empty.append(val_acc_no_empty)

            # Early Stopping Logic
            if patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0

                    if save_model and rank == 0:
                        # Save the absolute best model
                        best_path = os.path.join(save_dir, "best_model.pth")
                        state_dict = (
                            model.module.state_dict() if use_ddp else model.state_dict()
                        )
                        torch.save(state_dict, best_path)
                        print(f"Validation loss improved. Saved {best_path}")
                else:
                    epochs_no_improve += 1
                    if rank == 0:
                        print(
                            f"Val loss did not improve for {epochs_no_improve} epoch(s)."
                        )
                    if epochs_no_improve >= patience:
                        if rank == 0:
                            print(f"Early stopping triggered after {epoch + 1} epochs!")
                        break

    if plot_loss and rank == 0:
        if "SLURM_JOB_ID" in os.environ:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(epoch_train_losses, label="Train Loss")

            if val_loader is not None:
                plt.plot(epoch_val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Curve")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(epoch_train_acc_all, label="Train Acc (All)")
            plt.plot(epoch_train_acc_no_empty, label="Train Acc (Non-Empty)")
            if val_loader is not None:
                plt.plot(epoch_val_acc_all, label="Val Acc (All)")
                plt.plot(epoch_val_acc_no_empty, label="Val Acc (Non-Empty)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Accuracy Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "training_plot.png"))
        else:
            print("Cluster environment detected: Skipping plots.")


# ===============================================================
# MAIN EXECUTION
# ===============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train the Chess CNN")

    # --- Paths (Required) ---
    parser.add_argument(
        "--train_pkl",
        type=str,
        required=True,
        help="Path to the synthetic train data .pkl",
    )
    parser.add_argument(
        "--val_pkl", type=str, required=True, help="Path to the synthetic val data .pkl"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save the checkpoints"
    )

    # --- Hyperparameters (Optional with defaults) ---
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping")

    args = parser.parse_args()

    # --- DDP ---
    # Check for SLURM-Environment
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        use_ddp = True

        if rank == 0:
            print(f"DDP initialized via SLURM. Size: {world_size}")

    # Fallback with torch
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        use_ddp = True
        if rank == 0:
            print(f"DDP initialized via torchrun. Size: {world_size}")

    # Local no DDP
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_ddp = False
        print("No DDP environment found. Running in single process mode.")

    # Info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(local_rank)
        print(f"[{rank}/{world_size - 1}] Process on GPU: {gpu_name}")
    else:
        print(f"[{rank}/{world_size - 1}] Process on CPU")

    # Setup Config
    config = {
        "train_pickle_path": args.train_pkl,
        "cluster_train_pickle_path": args.train_pkl,
        "val_pickle_path": args.val_pkl,
        "cluster_val_pickle_path": args.val_pkl,
        "model_save_dir": args.save_dir,
        "cluster_model_save_dir": args.save_dir,
    }

    # Initialize Data and Model
    train_loader = get_train_loader(config, batch_size=args.batch_size, use_ddp=use_ddp)
    val_loader = get_val_loader(config, batch_size=args.batch_size, use_ddp=use_ddp)

    model = CustomChessCNN_v3(num_classes=13, dropout=0.3).to(device)

    # Wrap Model in DDP
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    # Start Training
    train(
        model=model,
        dataloader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_model=True,
        save_dir=args.save_dir,
        val_loader=val_loader,
        rank=rank,
        use_ddp=use_ddp,
        patience=args.patience,
    )

    # Cleanup
    if use_ddp:
        dist.destroy_process_group()
