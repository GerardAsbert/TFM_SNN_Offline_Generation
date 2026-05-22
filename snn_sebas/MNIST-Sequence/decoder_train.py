import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset_npz import SequentialMNISTDataset
from models import SNN, sequence_accuracy, sequence_loss
from snn_train import build_windowed_batch


SEED = 42
TRAIN_RATIO = 0.8
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "dataset" / "snn_global.pt"
DEFAULT_DECODER_PATH = Path(__file__).resolve().parent / "dataset" / "decoder_global.pt"


class LogitRefinerSNN(nn.Module):
    def __init__(
        self,
        input_size=10,
        hidden_size=64,
        output_size=10,
        neuron_type="mixed",
        threshold=1.0,
        tau=0.9,
        tau_o=0.0,
        tau_adapt=0.95,
        adapt_scale=0.2,
        lif_ratio=0.5,
        refractory_steps=0,
        feedback_type="symmetric",
        readout_mode="current",
        connectivity=1.0,
        tau_range=None,
        tau_adapt_range=None,
        threshold_range=None,
        adapt_scale_range=None,
    ):
        super().__init__()
        self.refiner = SNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            neuron_type=neuron_type,
            threshold=threshold,
            tau=tau,
            tau_o=tau_o,
            tau_adapt=tau_adapt,
            adapt_scale=adapt_scale,
            lif_ratio=lif_ratio,
            refractory_steps=refractory_steps,
            feedback_type=feedback_type,
            readout_mode=readout_mode,
            connectivity=connectivity,
            tau_range=tau_range,
            tau_adapt_range=tau_adapt_range,
            threshold_range=threshold_range,
            adapt_scale_range=adapt_scale_range,
        )
        self.mix = nn.Parameter(torch.tensor(0.5))

    def forward(self, base_logits, labels=None):
        refinement = self.refiner(base_logits, labels=labels)
        mix = torch.sigmoid(self.mix)
        return base_logits + mix * refinement


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--decoder-path", type=Path, default=DEFAULT_DECODER_PATH)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--decoder-hidden-size", type=int, default=64)
    parser.add_argument("--decoder-neuron-type", type=str, default="mixed", choices=["lif", "alif", "mixed"])
    parser.add_argument("--decoder-feedback-type", type=str, default="symmetric", choices=["random", "symmetric"])
    parser.add_argument("--decoder-readout-mode", type=str, default="current", choices=["filtered", "current"])
    parser.add_argument("--decoder-threshold", type=float, default=1.0)
    parser.add_argument("--decoder-tau", type=float, default=0.9)
    parser.add_argument("--decoder-tau-o", type=float, default=0.0)
    parser.add_argument("--decoder-tau-adapt", type=float, default=0.95)
    parser.add_argument("--decoder-adapt-scale", type=float, default=0.2)
    parser.add_argument("--decoder-lif-ratio", type=float, default=0.5)
    parser.add_argument("--decoder-refractory-steps", type=int, default=0)
    parser.add_argument("--decoder-conn", type=float, default=100.0)
    parser.add_argument("--decoder-tau-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None)
    parser.add_argument("--decoder-tau-adapt-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None)
    parser.add_argument("--decoder-threshold-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None)
    parser.add_argument("--decoder-adapt-scale-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_base_snn(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SNN(
        input_size=28 * checkpoint["window_width"],
        hidden_size=checkpoint["hidden_size"],
        neuron_type=checkpoint["neuron_type"],
        threshold=checkpoint.get("threshold", 1.0),
        tau=checkpoint["tau"],
        tau_o=checkpoint["tau_o"],
        tau_adapt=checkpoint["tau_adapt"],
        adapt_scale=checkpoint.get("adapt_scale", 0.2),
        lif_ratio=checkpoint["lif_ratio"],
        f_target=checkpoint["f_target"],
        c_reg=checkpoint["c_reg"],
        use_reg=checkpoint["use_reg"],
        refractory_steps=checkpoint["refractory_steps"],
        feedback_type=checkpoint["feedback_type"],
        readout_mode=checkpoint["readout_mode"],
        connectivity=checkpoint["connectivity"],
        tau_range=checkpoint.get("tau_range"),
        tau_adapt_range=checkpoint.get("tau_adapt_range"),
        threshold_range=checkpoint.get("threshold_range"),
        adapt_scale_range=checkpoint.get("adapt_scale_range"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def create_loaders(batch_size):
    dataset = SequentialMNISTDataset()
    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_refiner(args):
    connectivity = max(0.0, min(1.0, args.decoder_conn / 100.0))
    return LogitRefinerSNN(
        input_size=10,
        hidden_size=args.decoder_hidden_size,
        output_size=10,
        neuron_type=args.decoder_neuron_type,
        threshold=args.decoder_threshold,
        tau=args.decoder_tau,
        tau_o=args.decoder_tau_o,
        tau_adapt=args.decoder_tau_adapt,
        adapt_scale=args.decoder_adapt_scale,
        lif_ratio=args.decoder_lif_ratio,
        refractory_steps=args.decoder_refractory_steps,
        feedback_type=args.decoder_feedback_type,
        readout_mode=args.decoder_readout_mode,
        connectivity=connectivity,
        tau_range=args.decoder_tau_range,
        tau_adapt_range=args.decoder_tau_adapt_range,
        threshold_range=args.decoder_threshold_range,
        adapt_scale_range=args.decoder_adapt_scale_range,
    )


def run_epoch(base_model, refiner, loader, device, window_width, stride, optimizer=None):
    training = optimizer is not None
    refiner.train(training)
    total_loss = 0.0
    total_digit_acc = 0.0
    total_seq_acc = 0.0
    total_base_digit_acc = 0.0
    total_base_seq_acc = 0.0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x_seq, y_seq = build_windowed_batch(x, y, window_width, stride)

            with torch.inference_mode():
                base_logits = base_model(x_seq)

            if training:
                optimizer.zero_grad(set_to_none=True)

            refined_logits = refiner(base_logits, labels=y_seq if training else None)
            loss = sequence_loss(refined_logits, y_seq)

            if training:
                loss.backward()
                optimizer.step()

            digit_acc, seq_acc = sequence_accuracy(refined_logits, y_seq)
            base_digit_acc, base_seq_acc = sequence_accuracy(base_logits, y_seq)

            total_loss += loss.item()
            total_digit_acc += digit_acc
            total_seq_acc += seq_acc
            total_base_digit_acc += base_digit_acc
            total_base_seq_acc += base_seq_acc

    num_batches = len(loader)
    return {
        "loss": total_loss / num_batches,
        "digit_acc": total_digit_acc / num_batches,
        "seq_acc": total_seq_acc / num_batches,
        "base_digit_acc": total_base_digit_acc / num_batches,
        "base_seq_acc": total_base_seq_acc / num_batches,
    }


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model, checkpoint = load_base_snn(args.model_path, device)
    train_loader, val_loader = create_loaders(args.batch_size)
    refiner = build_refiner(args).to(device)
    optimizer = torch.optim.Adam(refiner.parameters(), lr=args.lr)

    window_width = checkpoint["window_width"]
    stride = checkpoint["stride"]
    best_val_seq_acc = -1.0
    best_epoch = 0
    stagnant_epochs = 0

    print(
        f"base_model={args.model_path.name} "
        f"window={window_width} stride={stride} "
        f"refiner=snn hidden={args.decoder_hidden_size} "
        f"neuron={args.decoder_neuron_type} "
        f"feedback={args.decoder_feedback_type} "
        f"readout={args.decoder_readout_mode}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            base_model, refiner, train_loader, device, window_width, stride, optimizer
        )
        val_metrics = run_epoch(
            base_model, refiner, val_loader, device, window_width, stride
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_digit_acc={train_metrics['digit_acc']:.3f} "
            f"train_seq_acc={train_metrics['seq_acc']:.3f} | "
            f"val_digit_acc={val_metrics['digit_acc']:.3f} "
            f"val_seq_acc={val_metrics['seq_acc']:.3f} | "
            f"base_val_digit_acc={val_metrics['base_digit_acc']:.3f} "
            f"base_val_seq_acc={val_metrics['base_seq_acc']:.3f}"
        )

        if val_metrics["seq_acc"] > best_val_seq_acc:
            best_val_seq_acc = val_metrics["seq_acc"]
            best_epoch = epoch
            stagnant_epochs = 0
            args.decoder_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "refiner_state_dict": refiner.state_dict(),
                    "decoder_hidden_size": args.decoder_hidden_size,
                    "decoder_neuron_type": args.decoder_neuron_type,
                    "decoder_feedback_type": args.decoder_feedback_type,
                    "decoder_readout_mode": args.decoder_readout_mode,
                    "decoder_threshold": args.decoder_threshold,
                    "decoder_tau": args.decoder_tau,
                    "decoder_tau_o": args.decoder_tau_o,
                    "decoder_tau_adapt": args.decoder_tau_adapt,
                    "decoder_adapt_scale": args.decoder_adapt_scale,
                    "decoder_lif_ratio": args.decoder_lif_ratio,
                    "decoder_refractory_steps": args.decoder_refractory_steps,
                    "decoder_conn": args.decoder_conn,
                    "decoder_tau_range": args.decoder_tau_range,
                    "decoder_tau_adapt_range": args.decoder_tau_adapt_range,
                    "decoder_threshold_range": args.decoder_threshold_range,
                    "decoder_adapt_scale_range": args.decoder_adapt_scale_range,
                    "base_model_path": str(args.model_path),
                    "window_width": window_width,
                    "stride": stride,
                    "best_val_seq_acc": best_val_seq_acc,
                    "best_epoch": best_epoch,
                },
                args.decoder_path,
            )
            print(f"Saved best decoder to {args.decoder_path}")
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= args.patience:
                print(
                    f"Early stopping after {epoch} epochs. "
                    f"Best val_seq_acc={best_val_seq_acc:.3f} at epoch {best_epoch}."
                )
                break

    print(f"Best val_seq_acc={best_val_seq_acc:.3f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
