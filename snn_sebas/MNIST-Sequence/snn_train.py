import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset_npz import SequentialMNISTDataset
from models import SNN, sequence_accuracy, sequence_loss


TRAIN_RATIO = 0.8
SEED = 42
MODEL_PATH = Path(__file__).resolve().parent / "dataset" / "snn_sequence.pt"
PRE_PLOTS_DIR = Path(__file__).resolve().parent / "dataset" / "snn_plots"
NUM_DIGITS = 5
DIGIT_WIDTH = 28

try:
    last_folder = max(int(child.name) for child in PRE_PLOTS_DIR.iterdir())
except:
    last_folder = -1
    
PLOTS_DIR = Path(__file__).resolve().parent / "dataset" / "snn_plots" / str(last_folder+1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=512)

    parser.add_argument("--window-width", type=int, default=28)
    parser.add_argument("--stride", type=int, default=28)

    parser.add_argument("--conn", type=float, default=70.0)
    parser.add_argument("--neuron-type", type=str, default="mixed", choices=["lif", "alif", "mixed"])
    parser.add_argument("--feedback-type", type=str, default="symmetric", choices=["random", "symmetric"])

    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--tau-o", type=float, default=0.0)
    parser.add_argument("--tau-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=(0.9, 0.995))
    parser.add_argument("--tau-adapt", type=float, default=0.95)
    parser.add_argument("--tau-adapt-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=(0.9, 0.98))

    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--threshold-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=(0.9, 1.1))

    parser.add_argument("--adapt-scale", type=float, default=0.2)
    parser.add_argument("--adapt-scale-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=(0.4, 0.6))

    parser.add_argument("--lif-ratio", type=float, default=0.45)
    parser.add_argument("--refractory-steps", type=int, default=0)

    parser.add_argument("--f-target", type=float, default=10.0)
    parser.add_argument("--c-reg", type=float, default=0.0)
    parser.add_argument("--use-reg", action="store_true")
    parser.add_argument("--readout-mode", type=str, default="current", choices=["filtered", "current"])

    parser.add_argument("--save-plots", action="store_true")

    return parser.parse_args()


def build_snn(input_size, args, connectivity):
    return SNN(
        input_size=input_size,
        hidden_size=args.hidden_size,
        neuron_type=args.neuron_type,
        threshold=args.threshold,
        tau=args.tau,
        tau_o=args.tau_o,
        tau_adapt=args.tau_adapt,
        adapt_scale=args.adapt_scale,
        lif_ratio=args.lif_ratio,
        refractory_steps=args.refractory_steps,
        feedback_type=args.feedback_type,
        readout_mode=args.readout_mode,
        f_target=args.f_target,
        c_reg=args.c_reg,
        use_reg=args.use_reg,
        connectivity=connectivity,
        tau_range=args.tau_range,
        tau_adapt_range=args.tau_adapt_range,
        threshold_range=args.threshold_range,
        adapt_scale_range=args.adapt_scale_range,
    )


def window_starts(total_width, window_width, stride):
    if window_width <= 0:
        raise ValueError("window_width must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if window_width > total_width:
        raise ValueError("window_width cannot exceed the image width")

    max_start = total_width - window_width
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def build_windowed_batch(images, labels, window_width, stride):
    total_width = images.shape[-1]
    starts = window_starts(total_width, window_width, stride)
    windows = []
    targets = []

    for start in starts:
        window = images[:, :, :, start : start + window_width]
        windows.append(window.reshape(window.shape[0], -1))

        center = start + (window_width // 2)
        digit_idx = min(center // DIGIT_WIDTH, NUM_DIGITS - 1)
        targets.append(labels[:, digit_idx])

    x_seq = torch.stack(windows, dim=1)
    y_seq = torch.stack(targets, dim=1)
    return x_seq, y_seq


def run_epoch(model, loader, device, window_width, stride, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_digit_acc = 0.0
    total_seq_acc = 0.0
    total_position_correct = None
    total_position_count = 0

    with torch.inference_mode() if not training else torch.enable_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x_seq, y_seq = build_windowed_batch(x, y, window_width, stride)

            if training:
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_seq, labels=y_seq)
                optimizer.step()
            else:
                logits = model(x_seq)

            loss = sequence_loss(logits, y_seq)
            digit_acc, seq_acc = sequence_accuracy(logits, y_seq)
            batch_correct = (logits.argmax(dim=-1) == y_seq).sum(dim=0).detach().cpu()
            if total_position_correct is None:
                total_position_correct = torch.zeros_like(batch_correct, dtype=torch.float64)
            total_position_correct += batch_correct.to(torch.float64)
            total_position_count += y_seq.shape[0]

            total_loss += loss.item()
            total_digit_acc += digit_acc
            total_seq_acc += seq_acc

    num_batches = len(loader)
    position_acc = (total_position_correct / max(1, total_position_count)).tolist()
    return (
        total_loss / num_batches,
        total_digit_acc / num_batches,
        total_seq_acc / num_batches,
        position_acc,
    )


def collect_predictions(model, loader, device, window_width, stride):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x_seq, y_seq = build_windowed_batch(x, y, window_width, stride)
            logits = model(x_seq)
            preds = logits.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_targets.append(y_seq.cpu())

    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


def save_history_plots(history, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_vs_epochs.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_seq_acc"], label="Train seq acc", linewidth=2)
    plt.plot(epochs, history["val_seq_acc"], label="Val seq acc", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Sequence accuracy")
    plt.title("Sequence Accuracy vs Epochs")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "sequence_accuracy_vs_epochs.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_digit_acc"], label="Train digit acc", linewidth=2)
    plt.plot(epochs, history["val_digit_acc"], label="Val digit acc", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Digit accuracy")
    plt.title("Digit Accuracy vs Epochs")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "digit_accuracy_vs_epochs.png", dpi=150)
    plt.close()


def save_evaluation_plots(preds, targets, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_np = preds.numpy()
    targets_np = targets.numpy()
    num_positions = targets_np.shape[1]

    position_acc = (preds_np == targets_np).mean(axis=0)
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(num_positions), position_acc, color="steelblue")
    plt.xticks(np.arange(num_positions), [f"Pos {i}" for i in range(num_positions)])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Position")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_per_position.png", dpi=150)
    plt.close()

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

    correct_digits_per_sequence = (preds_np == targets_np).sum(axis=1)
    bins = np.arange(num_positions + 2) - 0.5
    plt.figure(figsize=(8, 4))
    plt.hist(correct_digits_per_sequence, bins=bins, color="darkorange", rwidth=0.85)
    plt.xticks(range(num_positions + 1))
    plt.xlabel("Correct digits per sequence")
    plt.ylabel("Count")
    plt.title("Number of Correct Digits per Sequence")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "correct_digits_per_sequence.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    dataset = SequentialMNISTDataset()

    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 28 * args.window_width
    connectivity = max(0.0, min(1.0, args.conn / 100.0))
    model = build_snn(input_size, args, connectivity).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_digit_acc": [],
        "val_digit_acc": [],
        "train_seq_acc": [],
        "val_seq_acc": [],
    }

    best_val_seq_acc = -1.0
    model_path = MODEL_PATH.with_name("snn_global.pt")
    print(
        f"model=snn train={len(train_set)} "
        f"val={len(val_set)} batch={args.batch_size} "
        f"window={args.window_width} stride={args.stride} "
        f"neuron={args.neuron_type} feedback={args.feedback_type} "
        f"readout={args.readout_mode} "
        f"tau_range={args.tau_range or [args.tau, args.tau]} "
        f"tau_adapt_range={args.tau_adapt_range or [args.tau_adapt, args.tau_adapt]} "
        f"threshold_range={args.threshold_range or [args.threshold, args.threshold]} "
        f"adapt_scale_range={args.adapt_scale_range or [args.adapt_scale, args.adapt_scale]} "
        f"conn={args.conn:.1f}%"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_digit_acc, train_seq_acc, train_position_acc = run_epoch(
            model, train_loader, device, args.window_width, args.stride, optimizer
        )
        val_loss, val_digit_acc, val_seq_acc, val_position_acc = run_epoch(
            model, val_loader, device, args.window_width, args.stride
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_digit_acc"].append(train_digit_acc)
        history["val_digit_acc"].append(val_digit_acc)
        history["train_seq_acc"].append(train_seq_acc)
        history["val_seq_acc"].append(val_seq_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"train_digit_acc={train_digit_acc:.3f} "
            f"train_seq_acc={train_seq_acc:.3f} "
            f"train_pos={[round(v, 3) for v in train_position_acc]} | "
            f"val_loss={val_loss:.4f} "
            f"val_digit_acc={val_digit_acc:.3f} "
            f"val_seq_acc={val_seq_acc:.3f} "
            f"val_pos={[round(v, 3) for v in val_position_acc]}"
        )

        if val_seq_acc > best_val_seq_acc:
            best_val_seq_acc = val_seq_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_size": model.hidden_size,
                    "connectivity": connectivity,
                    "window_width": args.window_width,
                    "stride": args.stride,
                    "neuron_type": args.neuron_type,
                    "feedback_type": args.feedback_type,
                    "tau": args.tau,
                    "tau_range": args.tau_range,
                    "tau_o": args.tau_o,
                    "tau_adapt": args.tau_adapt,
                    "tau_adapt_range": args.tau_adapt_range,
                    "threshold": args.threshold,
                    "threshold_range": args.threshold_range,
                    "adapt_scale": args.adapt_scale,
                    "adapt_scale_range": args.adapt_scale_range,
                    "lif_ratio": args.lif_ratio,
                    "refractory_steps": args.refractory_steps,
                    "f_target": args.f_target,
                    "c_reg": args.c_reg,
                    "use_reg": args.use_reg,
                    "readout_mode": args.readout_mode,
                },
                model_path,
            )
            print(f"Saved best model to {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        reload_args = argparse.Namespace(**vars(args))
        reload_args.hidden_size = checkpoint.get("hidden_size", args.hidden_size)
        reload_args.neuron_type = checkpoint.get("neuron_type", args.neuron_type)
        reload_args.feedback_type = checkpoint.get("feedback_type", args.feedback_type)
        reload_args.readout_mode = checkpoint.get("readout_mode", args.readout_mode)
        reload_args.tau = checkpoint.get("tau", args.tau)
        reload_args.tau_range = checkpoint.get("tau_range", args.tau_range)
        reload_args.tau_o = checkpoint.get("tau_o", args.tau_o)
        reload_args.tau_adapt = checkpoint.get("tau_adapt", args.tau_adapt)
        reload_args.tau_adapt_range = checkpoint.get("tau_adapt_range", args.tau_adapt_range)
        reload_args.threshold = checkpoint.get("threshold", args.threshold)
        reload_args.threshold_range = checkpoint.get("threshold_range", args.threshold_range)
        reload_args.adapt_scale = checkpoint.get("adapt_scale", args.adapt_scale)
        reload_args.adapt_scale_range = checkpoint.get("adapt_scale_range", args.adapt_scale_range)
        reload_args.lif_ratio = checkpoint.get("lif_ratio", args.lif_ratio)
        reload_args.refractory_steps = checkpoint.get("refractory_steps", args.refractory_steps)
        reload_args.f_target = checkpoint.get("f_target", args.f_target)
        reload_args.c_reg = checkpoint.get("c_reg", args.c_reg)
        reload_args.use_reg = checkpoint.get("use_reg", args.use_reg)
        best_model = build_snn(
            input_size,
            reload_args,
            checkpoint.get("connectivity", connectivity),
        ).to(device)
        best_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        best_model = build_snn(input_size, args, connectivity).to(device)
        best_model.load_state_dict(checkpoint)
    preds, targets = collect_predictions(best_model, val_loader, device, args.window_width, args.stride)

    if args.save_plots:
        save_history_plots(history, PLOTS_DIR)
        save_evaluation_plots(preds, targets, PLOTS_DIR)
        print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()


# best: --epochs 20 --hidden-size 768 --conn 70 --lif-ratio 0.45 
# --adapt-scale-range 0.4 0.6 --tau-adapt-range 0.9 0.98 --tau-range 0.9 0.995 
# --threshold-range 0.9 1.1

# val_loss=0.2846 val_digit_acc=0.912 val_seq_acc=0.641 
# val_pos=[0.94, 0.902, 0.907, 0.907, 0.906]