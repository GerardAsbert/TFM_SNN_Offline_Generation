import csv
import re
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "snn_train.py"
RESULTS_PATH = SCRIPT_DIR / "dataset" / "snn_sweep_refined_results.csv"

EPOCH_RE = re.compile(
    r"Epoch\s+\d+\s+\|\s+train_loss=(?P<train_loss>[0-9.]+)\s+"
    r"train_digit_acc=(?P<train_digit_acc>[0-9.]+)\s+"
    r"train_seq_acc=(?P<train_seq_acc>[0-9.]+).*?\|\s+"
    r"val_loss=(?P<val_loss>[0-9.]+)\s+"
    r"val_digit_acc=(?P<val_digit_acc>[0-9.]+)\s+"
    r"val_seq_acc=(?P<val_seq_acc>[0-9.]+)"
)


def python_executable():
    venv_python = SCRIPT_DIR.parent / "venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def config_to_args(config):
    args = []
    for key, value in config.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(str(v) for v in value)
        else:
            args.extend([flag, str(value)])
    return args


def compact_label(config):
    return (
        f"h={config['hidden_size']} "
        f"ww={config['window_width']}/st={config['stride']} "
        f"lif={config['lif_ratio']} "
        f"tau={tuple(config['tau_range'])} "
        f"ta={tuple(config['tau_adapt_range'])} "
        f"th={tuple(config['threshold_range'])} "
        f"ad={tuple(config['adapt_scale_range'])}"
    )


def run_config(config):
    command = [python_executable(), str(TRAIN_SCRIPT), *config_to_args(config)]
    result = subprocess.run(
        command,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    matches = EPOCH_RE.findall(result.stdout)
    if not matches:
        return {
            **config,
            "returncode": result.returncode,
            "status": "failed",
            "train_loss": "",
            "train_digit_acc": "",
            "train_seq_acc": "",
            "val_loss": "",
            "val_digit_acc": "",
            "val_seq_acc": "",
        }, result.stdout, result.stderr

    train_loss, train_digit_acc, train_seq_acc, val_loss, val_digit_acc, val_seq_acc = matches[-1]
    return {
        **config,
        "returncode": result.returncode,
        "status": "ok" if result.returncode == 0 else "failed",
        "train_loss": train_loss,
        "train_digit_acc": train_digit_acc,
        "train_seq_acc": train_seq_acc,
        "val_loss": val_loss,
        "val_digit_acc": val_digit_acc,
        "val_seq_acc": val_seq_acc,
    }, result.stdout, result.stderr


def refined_configs():
    base = {
        "epochs": 16,
        "lr": 1e-3,
        "neuron_type": "mixed",
        "feedback_type": "symmetric",
        "readout_mode": "current",
        "tau_o": 0.0,
        "conn": 100.0,
    }

    configs = []

    for hidden_size in [192, 224]:
        for lif_ratio in [0.45, 0.5, 0.7]:
            for tau_range in [(0.88, 0.99), (0.90, 0.995)]:
                for tau_adapt_range in [(0.90, 0.98), (0.94, 0.995)]:
                    for threshold_range in [(0.9, 1.1), (0.9, 1.15)]:
                        for adapt_scale_range in [(0.2, 0.5), (0.4, 0.6)]:
                            configs.append(
                                {
                                    **base,
                                    "window_width": 28,
                                    "stride": 28,
                                    "hidden_size": hidden_size,
                                    "lif_ratio": lif_ratio,
                                    "tau_range": tau_range,
                                    "tau_adapt_range": tau_adapt_range,
                                    "threshold_range": threshold_range,
                                    "adapt_scale_range": adapt_scale_range,
                                }
                            )

    for hidden_size in [192]:
        for lif_ratio in [0.5]:
            for tau_range in [(0.90, 0.995)]:
                for tau_adapt_range in [(0.94, 0.995)]:
                    for threshold_range in [(0.9, 1.1)]:
                        for adapt_scale_range in [(0.2, 0.5)]:
                            configs.append(
                                {
                                    **base,
                                    "window_width": 8,
                                    "stride": 4,
                                    "hidden_size": hidden_size,
                                    "lif_ratio": lif_ratio,
                                    "tau_range": tau_range,
                                    "tau_adapt_range": tau_adapt_range,
                                    "threshold_range": threshold_range,
                                    "adapt_scale_range": adapt_scale_range,
                                }
                            )

    return configs


def save_results(rows):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows, limit=12):
    header = f"{'#':>3}  {'val_dig':>7}  {'val_seq':>7}  {'ww/st':>7}  config"
    print(header)
    print("-" * len(header))
    for index, row in enumerate(rows[:limit], start=1):
        val_dig = f"{float(row['val_digit_acc']):.3f}" if row.get("val_digit_acc") else "n/a"
        val_seq = f"{float(row['val_seq_acc']):.3f}" if row.get("val_seq_acc") else "n/a"
        wwst = f"{row['window_width']}/{row['stride']}"
        print(f"{index:>3}  {val_dig:>7}  {val_seq:>7}  {wwst:>7}  {compact_label(row)}")


def main():
    configs = refined_configs()
    rows = []
    print(f"Running {len(configs)} refined sweep configs")
    print()

    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {compact_label(config)}")
        row, stdout, stderr = run_config(config)
        rows.append(row)
        if row["status"] == "ok":
            print(
                f"        val_digit_acc={float(row['val_digit_acc']):.3f} "
                f"val_seq_acc={float(row['val_seq_acc']):.3f}"
            )
        else:
            print("        failed")
            print(stdout)
            print(stderr)

    rows.sort(
        key=lambda row: (
            float(row["val_digit_acc"]) if row.get("val_digit_acc") not in {"", None} else float("-inf"),
            float(row["val_seq_acc"]) if row.get("val_seq_acc") not in {"", None} else float("-inf"),
        ),
        reverse=True,
    )

    save_results(rows)
    print()
    print("Top results")
    print_table(rows)
    print()
    print(f"saved results to {RESULTS_PATH}")
    if rows:
        print(f"best config: {compact_label(rows[0])}")


if __name__ == "__main__":
    main()
