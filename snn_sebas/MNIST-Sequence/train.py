import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset_npz import SequentialMNISTDataset
from models import build_model, sequence_accuracy, sequence_loss


BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.8
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["cnn", "crnn", "tcn"])
    return parser.parse_args()


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_digit_acc = 0.0
    total_seq_acc = 0.0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = sequence_loss(logits, y)

            if training:
                loss.backward()
                optimizer.step()

            digit_acc, seq_acc = sequence_accuracy(logits, y)
            total_loss += loss.item()
            total_digit_acc += digit_acc
            total_seq_acc += seq_acc

    num_batches = len(loader)
    return (
        total_loss / num_batches,
        total_digit_acc / num_batches,
        total_seq_acc / num_batches,
    )


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

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model_path = Path(__file__).resolve().parent / "dataset" / f"{args.model}_baseline.pt"

    best_val_seq_acc = -1.0
    print(
        f"model={args.model} train={len(train_set)} "
        f"val={len(val_set)} batch={BATCH_SIZE}"
    )

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_digit_acc, train_seq_acc = run_epoch(
            model, train_loader, device, optimizer
        )
        val_loss, val_digit_acc, val_seq_acc = run_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"train_digit_acc={train_digit_acc:.3f} "
            f"train_seq_acc={train_seq_acc:.3f} | "
            f"val_loss={val_loss:.4f} "
            f"val_digit_acc={val_digit_acc:.3f} "
            f"val_seq_acc={val_seq_acc:.3f}"
        )

        if val_seq_acc > best_val_seq_acc:
            best_val_seq_acc = val_seq_acc
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")


if __name__ == "__main__":
    main()
