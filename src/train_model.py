import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from evaluation.eval import evaluate_rotation_invariant
from utils.models import CustomChessCNN_v3
from utils.dataset import (
    get_path_from_config_file,
    get_train_loader,
    get_val_loader,
)
from tqdm import tqdm


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
    rank=0,  # <- FLAG: RANK
    use_ddp=False,  # <- FLAG: DDP
):
    if not use_ddp:
        model = model.to(device)

    weights = torch.ones(13)
    weights[empty_class_idx] = empty_class_weight
    weights = weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_train_losses = []
    epoch_train_acc_all = []
    epoch_train_acc_no_empty = []

    epoch_val_losses = []
    epoch_val_acc_all = []
    epoch_val_acc_no_empty = []

    if save_model and rank == 0:
        assert (
            save_dir is not None
        ), "Please provide a save_dir to save model checkpoints."
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        if use_ddp:
            dataloader.sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        correct_all = 0
        correct_non_empty = 0
        total_all = 0
        total_non_empty = 0

        # TQDM Bar nur auf Rank 0 anzeigen, sonst crasht der Log
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
            print(f"✅ Model saved to: {save_path}")

        # --- Evaluate ---
        if val_loader is not None:
            val_loss, val_acc_all, val_acc_no_empty, _ = evaluate_rotation_invariant(
                model, val_loader, device=device, rank=rank
            )
            epoch_val_losses.append(val_loss)
            epoch_val_acc_all.append(val_acc_all)
            epoch_val_acc_no_empty.append(val_acc_no_empty)

    if plot_loss and rank == 0:
        if "SLURM_JOB_ID" in os.environ:
            plt.figure(figsize=(12, 5))
            # Loss plot
            plt.subplot(1, 2, 1)
            plt.plot(epoch_train_losses, label="Train Loss")
            if val_loader is not None:
                plt.plot(epoch_val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Curve")
            plt.legend()
            # Accuracy plot
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
            # plt.show()
        else:
            print("Cluster environment detected: Skipping plots.")


if __name__ == "__main__":
    # --- DDP INITIALISIERUNG ---

    # SLURM DDP
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])

        # WICHTIG: PyTorch interne Variablen setzen, falls Bibliotheken sich darauf verlassen
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

        # Initialisiere Prozess-Gruppe
        # (MASTER_ADDR und MASTER_PORT müssen im submit_job.sh exportiert sein!)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        use_ddp = True

        if rank == 0:
            print(f"🚀 DDP initialized via SLURM. World size: {world_size}")

    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        use_ddp = True
        if rank == 0:
            print(f"🚀 DDP initialized via torchrun. World size: {world_size}")

    # No DDP (for local testing)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_ddp = False
        print("⚠️ No DDP environment found. Running in single process mode.")

    # Info-Ausgabe
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(local_rank)
        print(f"[{rank}/{world_size - 1}] Process on GPU: {gpu_name}")
    else:
        print(f"[{rank}/{world_size - 1}] Process on CPU")

    # Load Configuration
    with open(r"/home/vihps/vihps01/DLCV_Chess/config.json", "r") as f:
        config = json.load(f)

    save_dir = get_path_from_config_file(config, "model_save_dir")

    # Initialize Data and Model
    train_loader = get_train_loader(config, batch_size=16, use_ddp=use_ddp)
    val_loader = get_val_loader(config, batch_size=16, use_ddp=use_ddp)

    model = CustomChessCNN_v3(num_classes=13, dropout=0.3).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    train(
        model=model,
        dataloader=train_loader,
        device=device,
        epochs=35,
        save_model=True,
        save_dir=save_dir,
        val_loader=val_loader,
        rank=rank,
        use_ddp=use_ddp,
    )

    # Cleanup
    if use_ddp:
        dist.destroy_process_group()
