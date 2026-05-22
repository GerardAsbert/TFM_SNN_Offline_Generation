import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from loader import NMNISTDataset
from modelsV6 import SNN



TRAIN_DIR = "468j46mzdv-1/Train"
TEST_DIR = "468j46mzdv-1/Test"
INPUT_SIZE : int = 2312 

MODEL_PATH = Path(__file__).resolve().parent / "snn_best_model.pt"
PRE_PLOTS_DIR = Path(__file__).resolve().parent / "snn_plots"

try:
    last_folder = max(int(child.name) for child in PRE_PLOTS_DIR.iterdir())
except:
    last_folder = -1
    
PLOTS_DIR = Path(__file__).resolve().parent / "snn_plots" / str(last_folder+1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--tau-o", type=float, default=0.9)
    parser.add_argument("--f-target", type=float, default=10.0)
    parser.add_argument("--c-reg", type=float, default=0.0)
    parser.add_argument("--use-reg", action="store_true")
    parser.add_argument("--refractory-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--plot-dir", type=str, default="plots_v6")
    parser.add_argument("--plot-sample-index", type=int, default=0)
    parser.add_argument("--plot-neuron-idx", type=int, default=0)
    return parser.parse_args()


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    use_non_blocking = loader.pin_memory and device.type == "cuda"

    for x, y in tqdm(loader, desc="Training"):
        x = x.to(device, non_blocking=use_non_blocking)
        y = y.to(device, non_blocking=use_non_blocking)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, labels=y)
        loss = loss_fn(logits, y)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    correct = 0
    use_non_blocking = loader.pin_memory and device.type == "cuda"

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=use_non_blocking)
            y = y.to(device, non_blocking=use_non_blocking)
            correct += (model(x).argmax(1) == y).sum().item()

    return correct / len(loader.dataset)


def collect_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_targets.append(y.cpu())

    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


def save_history_plots(history, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
    #plt.plot(epochs, history["val_loss"], label="Val loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_vs_epochs.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_acc"], label="Train acc", linewidth=2)
    plt.plot(epochs, history["test_acc"], label="Test acc", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_epochs.png", dpi=150)
    plt.close()


def save_evaluation_plots(preds, targets, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_np = preds.numpy()
    targets_np = targets.numpy()

    confusion = np.zeros((10, 10), dtype=np.int64)
    flat_targets = targets_np.reshape(-1)
    flat_preds = preds_np.reshape(-1)
    for true_digit, pred_digit in zip(flat_targets, flat_preds):
        confusion[true_digit, pred_digit] += 1

    plt.figure(figsize=(7, 6))
    plt.imshow(confusion, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted digit")
    plt.ylabel("True digit")
    plt.title("Confusion Matrix")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pin_memory = args.pin_memory or device.type == "cuda"
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        NMNISTDataset(
            TRAIN_DIR,
            seq_len=args.seq_len,
        ),
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        NMNISTDataset(
            TEST_DIR,
            seq_len=args.seq_len,
        ),
        shuffle=False,
        **loader_kwargs,
    )

    model = SNN(
        i_size=INPUT_SIZE,
        h_size=args.hidden_size,
        tau=args.tau,
        tau_o=args.tau_o,
        f_target=args.f_target,
        c_reg=args.c_reg,
        use_reg=args.use_reg,
        refractory_steps=args.refractory_steps,
    ).to(device)


    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}


    best_test_acc = -1.0
    model_path = MODEL_PATH.with_name("snn.pt")


    print(
        "Config | "
        f"hidden={args.hidden_size} "
        f"use_reg={args.use_reg} "
        f"c_reg={args.c_reg} "
        f"refractory_steps={args.refractory_steps} "
        f"seq_len={args.seq_len} "
        f"batch={args.batch_size} "
        f"lr={args.lr} "
    )

    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        test_acc = eval_epoch(model, test_loader, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(
            f"Epoch {epoch:02d} | Loss: {loss:.4f} | "
            f"Train: {train_acc:.3f} | Test: {test_acc:.3f}"
        )


        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_size": model.hidden_size,
                },
                model_path,
            )
            print(f"Saved best model to {model_path}")

    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        best_model = SNN(
            INPUT_SIZE,
            checkpoint.get("hidden_size", args.hidden_size),
        ).to(device)
        best_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        best_model = SNN(INPUT_SIZE, args.hidden_size).to(device)
        best_model.load_state_dict(checkpoint)
    preds, targets = collect_predictions(best_model, test_loader, device)

    save_history_plots(history, PLOTS_DIR)
    save_evaluation_plots(preds, targets, PLOTS_DIR)
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
