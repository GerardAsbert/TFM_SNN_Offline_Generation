"""
train.py — PyTorch e-prop handwriting generation.

Mirrors the pipeline of nest_handwriting_eprop_with_pen.py but uses PyTorch.

Modalities
----------
1. Style variation  – base + jitter-style spike encoding, multiple styles per letter
3. Alphabet generation – one frozen spike pattern per letter
"""

import argparse
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pickle
import numpy as np
import torch
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import wandb

from models import HandwritingSNN
from grad_health import log_gradient_health


# ── reproducibility ────────────────────────────────────────────────────────────
rng_seed = 27
np.random.seed(rng_seed)
torch.manual_seed(rng_seed)


# ── sweep defaults ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "threshold":            1.0,
    "w_gain":               1.0,
    "lr":                   5e-3,
    "gamma":                0.3,
    "c_reg":                0,
    "n_rec":                200,
    "tau_a_ms":             2000,
    "prob":                 0.05,
    "learning_signal_mode": "random",
}


# ── dataset helpers ────────────────────────────────────────────────────────────

def load_dataset(dataset_path: str, n_letras: int):
    """Load n_letras trajectory .txt files. Returns (datos_letras, trayectorias)."""
    trayectorias = sorted(f for f in os.listdir(dataset_path) if f.endswith(".txt"))
    if n_letras > len(trayectorias):
        raise ValueError(
            f"Only {len(trayectorias)} trajectories in {dataset_path}, "
            f"but {n_letras} were requested."
        )
    datos_letras = []
    for i in range(n_letras):
        datos = np.loadtxt(os.path.join(dataset_path, trayectorias[i]))
        if datos.ndim != 2 or datos.shape[1] not in (2, 3):
            raise ValueError(f"{trayectorias[i]} must have 2 or 3 columns (dx, dy[, pen])")
        if datos.shape[1] == 2:
            datos = np.hstack([datos, np.ones((datos.shape[0], 1), dtype=datos.dtype)])
        datos_letras.append(datos)
    return datos_letras, trayectorias


def load_dataset2(dataset_path: str):
    """
    Load dataset with structure:
        dataset_path/<author>/<symbol>/<instance>.txt

    Returns:
        data    : dict {author_idx: {symbol_idx: [np.ndarray, ...]}}
        authors : sorted list of author directory names
        symbols : sorted list of symbol directory names (union across all authors)
    """
    authors = sorted(
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    )
    if not authors:
        raise ValueError(f"No author directories found in {dataset_path}")

    all_symbols: set = set()
    for author in authors:
        author_path = os.path.join(dataset_path, author)
        all_symbols.update(
            d for d in os.listdir(author_path)
            if os.path.isdir(os.path.join(author_path, d))
        )
    symbols = sorted(all_symbols)

    data: dict = {}
    for ai, author in enumerate(authors):
        data[ai] = {}
        author_path = os.path.join(dataset_path, author)
        for si, symbol in enumerate(symbols):
            symbol_path = os.path.join(author_path, symbol)
            if not os.path.isdir(symbol_path):
                data[ai][si] = []
                continue
            txt_files = sorted(f for f in os.listdir(symbol_path) if f.endswith(".txt"))
            instances = []
            for fname in txt_files:
                arr = np.loadtxt(os.path.join(symbol_path, fname))
                if arr.ndim != 2 or arr.shape[1] not in (2, 3):
                    raise ValueError(f"{fname} must have 2 or 3 columns (dx, dy[, pen])")
                if arr.shape[1] == 2:
                    arr = np.hstack([arr, np.ones((arr.shape[0], 1), dtype=arr.dtype)])
                instances.append(arr)
            data[ai][si] = instances

    return data, authors, symbols


def build_targets(
    datos_letras,
    n_sequences: int,
    seq_T: int,
    data_point: int,
    traj_idx_per_seq,   # (n_sequences,) int array: index into datos_letras
) -> np.ndarray:
    """
    Interpolate each trajectory to seq_T timesteps and compute cumulative
    (dx, dy, pen) targets. Returns float32 array of shape (n_sequences, seq_T, 3).
    """
    x_eval = np.arange(seq_T) / data_point
    x_data = np.arange(seq_T // data_point)

    targets = np.empty((n_sequences, seq_T, 3), dtype=np.float32)
    for i in range(n_sequences):
        raw = datos_letras[int(traj_idx_per_seq[i])]
        dx_c = np.cumsum(raw[:, 0].astype(float))
        dy_c = np.cumsum(raw[:, 1].astype(float))
        if np.max(np.abs(dx_c)) > 0:
            dx_c /= np.max(np.abs(dx_c))
        dx_c -= dx_c[0]
        if np.max(np.abs(dy_c)) > 0:
            dy_c /= np.max(np.abs(dy_c))
        dy_c -= dy_c[0]
        pen = raw[:, 2].astype(float) if raw.shape[1] >= 3 else np.ones(raw.shape[0])
        dx_pos = np.interp(x_eval, x_data, dx_c)
        dy_pos = np.interp(x_eval, x_data, dy_c)
        targets[i, :, 0] = np.diff(dx_pos, prepend=dx_pos[0]).astype(np.float32)
        targets[i, :, 1] = np.diff(dy_pos, prepend=dy_pos[0]).astype(np.float32)
        targets[i, :, 2] = np.clip(np.interp(x_eval, x_data, pen), 0.0, 1.0).astype(np.float32)
    return targets


# ── spike-train generators ─────────────────────────────────────────────────────

def generate_spikes_modal3(
    n_in: int,
    seq_T: int,
    n_letras: int,
    prob: float = 0.05,
):
    """
    Generate ONE frozen spike train per letter.
    Returns a dictionary:
        spikes_por_letra[letter_idx] -> (seq_T, n_in)
    """

    spikes_por_letra = {}

    for li in tqdm(range(n_letras), desc="Generating spikes per character"):
        rng = np.random.RandomState(1000 + li)

        s = (rng.rand(seq_T, n_in) < prob).astype(np.float32)

        s[:, 0] = 0

        spikes_por_letra[li] = s

    return spikes_por_letra

def generate_spikes_character_and_style(
    n_in: int,
    seq_T: int,
    n_letras: int,
    n_authors: int,
    data: dict,          # data[author_idx][symbol_idx] = [array, ...]
    prob: float = 0.05,
    jitter_window: int = 10,
) -> dict:
    """
    Build spike trains for the character+style modality.

    Each full spike train (seq_T, n_in) is the concatenation of:
      - first half  (seq_T//2, n_in): frozen character pattern for that symbol
      - second half (seq_T//2, n_in): base or jittered style pattern for that author

    For each (author, symbol) pair with N instances:
      instance 0       -> base style half (no jitter)
      instance 1..N-1  -> jittered versions of the base style half

    Returns dict keyed by (author_idx, symbol_idx, instance_idx) -> (seq_T, n_in) float32.
    """
    half_T = seq_T // 2

    char_spikes = {}
    for li in tqdm(range(n_letras), desc="Generating character spikes"):
        rng = np.random.RandomState(li)
        s = (rng.rand(half_T, n_in) < prob).astype(np.float32)
        s[:, 0] = 0
        char_spikes[li] = s  # (half_T, n_in)

    style_base = {}
    for ai in tqdm(range(n_authors), desc="Generating style base spikes"):
        rng = np.random.RandomState(1000 + ai)
        s = (rng.rand(half_T, n_in) < prob).astype(np.float32)
        s[:, 0] = 0
        style_base[ai] = s  # (half_T, n_in)

    spikes = {}
    for ai in range(n_authors):
        for si in range(n_letras):
            instances = data[ai].get(si, [])
            for inst_idx in range(len(instances)):
                char_half = char_spikes[si]
                if inst_idx == 0:
                    style_half = style_base[ai]
                else:
                    # _jitter_spike_train expects (n_neu, T); style_base is (half_T, n_in)
                    np.random.seed(2000 + ai * 100_000 + si * 1000 + inst_idx)
                    style_half = _jitter_spike_train(
                        style_base[ai].T, window=jitter_window
                    ).T.astype(np.float32)  # (half_T, n_in)
                    style_half[:, 0] = 0
                spikes[(ai, si, inst_idx)] = np.concatenate(
                    [char_half, style_half], axis=0  # (seq_T, n_in)
                )

    return spikes


def _jitter_spike_train(spikes: np.ndarray, window: int = 10) -> np.ndarray:
    """Row-wise random temporal jitter of a (n_neu, T) binary spike matrix."""
    n_neu, T = spikes.shape
    out = np.zeros_like(spikes, dtype=bool)
    for i in range(n_neu):
        idx = np.flatnonzero(spikes[i])
        if idx.size == 0:
            continue
        shifts = np.random.randint(-window, window + 1, size=idx.size)
        out[i, np.clip(idx + shifts, 0, T - 1)] = True
    out[:, 0] = False
    return out


def generate_spikes_modal1(
    n_base: int,
    n_style: int,
    n_letras_distintas: int,
    n_estilos_por_letra: int,
    n_sequences: int,
    seq_T: int,
    letras_estilos_por_secuencia,   # (n_sequences, 2): columns = [letra_idx, estilo_idx]
    prob: float = 0.05,
    jitter_window: int = 10,
) -> np.ndarray:
    """
    Modality 1: base spike trains (letter-specific) concatenated with
    style spike trains (jittered from style-0). Matches NEST modality-1 encoding.
    n_in = n_base + n_style.
    Returns (n_sequences, seq_T, n_in) float32.
    """
    n_in = n_base + n_style

    # one frozen base pattern per letter
    spikes_base = {}
    for li in range(n_letras_distintas):
        np.random.seed(42 + li)
        s = (np.random.rand(n_base, seq_T) < prob)
        s[0, :] = 0
        spikes_base[li] = s

    # style 0 random; style k>0 = temporal jitter of style 0
    spikes_style = {}
    for li in range(n_letras_distintas):
        np.random.seed(100 + li)
        s0 = (np.random.rand(n_style, seq_T) < prob)
        s0[0, :] = 0
        spikes_style[li * n_estilos_por_letra] = s0
        for si in range(1, n_estilos_por_letra):
            np.random.seed(200 + li + 1000 * si)
            sj = _jitter_spike_train(s0, window=jitter_window)
            sj[0, :] = 0
            spikes_style[li * n_estilos_por_letra + si] = sj

    result = np.empty((n_sequences, seq_T, n_in), dtype=np.float32)
    for i in range(n_sequences):
        li = int(letras_estilos_por_secuencia[i, 0])
        si = int(letras_estilos_por_secuencia[i, 1])
        combined = np.vstack([spikes_base[li], spikes_style[li * n_estilos_por_letra + si]])
        result[i] = combined.T.astype(np.float32)   # (seq_T, n_in)
    return result

def _to_positions(arr):
    """Integrate velocity channels (0,1) back to absolute positions via cumulative
    sum along time. Channel 2 (pen) is left unchanged. Works for (..., T, 3)."""
    out = arr.copy()
    out[..., 0] = np.cumsum(arr[..., 0], axis=-1)
    out[..., 1] = np.cumsum(arr[..., 1], axis=-1)
    return out

# ── analysis and plots ─────────────────────────────────────────────────────────

def _segments_from_mask(x, y, mask):
    """Split arrays into contiguous pen-down segments."""
    if isinstance(mask, slice):
        return [(x, y)]
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    starts = np.where(~m[:-1] & m[1:])[0] + 1
    ends   = np.where( m[:-1] & ~m[1:])[0] + 1
    if m[0]:
        starts = np.r_[0, starts]
    if m[-1]:
        ends = np.r_[ends, m.size]
    return [(x[s:e], y[s:e]) for s, e in zip(starts, ends) if e > s]


def analyze_and_plot(
    outputs,                # (n_sequences, seq_T, 3)
    targets,                # (n_sequences, seq_T, 3)
    traj_idx_per_seq,       # (n_sequences,) — groups sequences by trajectory
    trayectorias,
    output_dir: str,
    loss_history=None,
):
    """Post-training analysis: loss curve, per-letter MSE, trajectory plots."""
    os.makedirs(output_dir, exist_ok=True)

    outputs = _to_positions(outputs)
    targets = _to_positions(targets)

    if loss_history:
        fig_loss = plt.figure()
        plt.plot(range(1, len(loss_history) + 1), loss_history)
        plt.xlabel("training iteration")
        plt.ylabel("MSE loss")
        plt.title("Training loss")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_training.png", dpi=300)
        wandb.log({"charts/loss_curve": wandb.Image(fig_loss)})
        plt.close()

    t_pen_all = targets[:, :, 2].ravel()
    m_global = t_pen_all >= 0.5

    global_metrics = {}
    for ch, name in enumerate(["dx", "dy", "pen"]):
        y_c = outputs[:, :, ch].ravel()
        t_c = targets[:, :, ch].ravel()
        sel = m_global if (ch < 2 and np.any(m_global)) else slice(None)
        mse_val = mean_squared_error(t_c[sel], y_c[sel])
        print(f"MSE {name}: {mse_val:.6f}")
        global_metrics[f"val/mse_{name}"] = mse_val
    wandb.log(global_metrics)

    letras_unicas = sorted(np.unique(traj_idx_per_seq))

    for li in letras_unicas:
        nombre_base = os.path.splitext(trayectorias[li])[0]
        idxs = np.where(traj_idx_per_seq == li)[0]
        last_i = idxs[-1]

        r_dx  = outputs[last_i, :, 0]
        r_dy  = -outputs[last_i, :, 1]
        r_pen = outputs[last_i, :, 2]
        t_dx  = targets[last_i, :, 0]
        t_dy  = -targets[last_i, :, 1]
        t_pen = targets[last_i, :, 2]

        mR  = r_pen >= 0.5
        mT  = t_pen >= 0.5
        m_e = mT if np.any(mT) else slice(None)

        mse_x = mean_squared_error(t_dx[m_e], r_dx[m_e])
        mse_y = mean_squared_error(t_dy[m_e], r_dy[m_e])
        mse_avg = (mse_x + mse_y) / 2
        print(f"  {nombre_base}: MSE_X={mse_x:.6f}  MSE_Y={mse_y:.6f}  avg={mse_avg:.6f}")
        wandb.log({
            f"val/mse_x_{nombre_base}": mse_x,
            f"val/mse_y_{nombre_base}": mse_y,
            f"val/mse_avg_{nombre_base}": mse_avg,
        })

        fig, ax = plt.subplots(figsize=(6, 3))
        for xs, ys in _segments_from_mask(t_dx, t_dy, mT):
            ax.plot(xs, ys, color="tab:blue", label="target", alpha=0.85)
        for xs, ys in _segments_from_mask(r_dx, r_dy, mR):
            ax.plot(xs, ys, color="tab:red", label="readout", linewidth=1)
        ax.set_title(f"{nombre_base}  MSE X:{mse_x:.4f} Y:{mse_y:.4f}")
        ax.axis("equal")
        ax.axis("off")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            dict(zip(labels, handles)).values(),
            dict(zip(labels, handles)).keys(),
            loc="lower right", fontsize=8,
        )
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{nombre_base}.png", dpi=300)
        plt.close()

    # mosaic of all letters (pen-down strokes only)
    if len(letras_unicas) > 1:
        espacio = 1.2
        fig, ax = plt.subplots(figsize=(max(6, 4 * len(letras_unicas)), 4))
        for col, li in enumerate(letras_unicas):
            nombre_base = os.path.splitext(trayectorias[li])[0]
            idxs = np.where(traj_idx_per_seq == li)[0]
            last_i = idxs[-1]
            offset = col * espacio
            r_dx  = outputs[last_i, :, 0] + offset
            r_dy  = -outputs[last_i, :, 1]
            t_dx  = targets[last_i, :, 0] + offset
            t_dy  = -targets[last_i, :, 1]
            mR = outputs[last_i, :, 2] >= 0.5
            mT = targets[last_i, :, 2] >= 0.5
            for xs, ys in _segments_from_mask(t_dx, t_dy, mT):
                ax.plot(xs, ys, color="blue", alpha=0.6)
            for xs, ys in _segments_from_mask(r_dx, r_dy, mR):
                ax.plot(xs, ys, color="red", alpha=0.85)
            try:
                label = chr(int(nombre_base))
            except Exception:
                label = nombre_base
            ax.text(offset, 0.0, label, fontsize=10, weight="bold", ha="center", va="top")
        ax.set_title("Mosaic (pen-down strokes)")
        ax.axis("equal")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mosaic.png", dpi=300)
        wandb.log({"images/mosaic": wandb.Image(fig)})
        plt.close()


# ── periodic image logging helper ─────────────────────────────────────────────

def _log_character_images_to_wandb(
    model, spikes_torch, letras_por_secuencia, targets_torch, trayectorias, step
):
    """Run inference and log one trajectory image per unique character to wandb."""
    letras_unicas = sorted(set(int(l) for l in letras_por_secuencia))

    with torch.no_grad():
        for li in letras_unicas:
            nombre_base = os.path.splitext(trayectorias[li])[0]
            idxs = np.where(np.array(letras_por_secuencia) == li)[0]
            last_i = idxs[-1]

            x = spikes_torch[li].unsqueeze(0)         # (1, T, n_in)
            out = _to_positions(model(x).cpu().numpy()[0])   # (T, 3) -> positions
            tgt = _to_positions(targets_torch[last_i].cpu().numpy())

            r_dx  = out[:, 0]; r_dy = -out[:, 1]; r_pen = out[:, 2]
            t_dx  = tgt[:, 0]; t_dy = -tgt[:, 1]; t_pen = tgt[:, 2]

            mR = r_pen >= 0.5
            mT = t_pen >= 0.5
            m_e = mT if np.any(mT) else slice(None)

            mse_x = mean_squared_error(t_dx[m_e], r_dx[m_e])
            mse_y = mean_squared_error(t_dy[m_e], r_dy[m_e])

            fig, ax = plt.subplots(figsize=(6, 3))
            for xs, ys in _segments_from_mask(t_dx, t_dy, mT):
                ax.plot(xs, ys, color="tab:blue", label="target", alpha=0.85)
            for xs, ys in _segments_from_mask(r_dx, r_dy, mR):
                ax.plot(xs, ys, color="tab:red", label="readout", linewidth=1)
            ax.set_title(f"{nombre_base}  iter {step}  MSE X:{mse_x:.4f} Y:{mse_y:.4f}")
            ax.axis("equal")
            ax.axis("off")
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                dict(zip(labels, handles)).values(),
                dict(zip(labels, handles)).keys(),
                loc="lower right", fontsize=8,
            )
            plt.tight_layout()
            wandb.log({f"images/{nombre_base}": wandb.Image(fig)})
            plt.close()


# ── core training loop ─────────────────────────────────────────────────────────

def run_training(
    spikes_dict,
    targets_np,     # (n_sequences, seq_T, 3)
    letras_ids,
    n_iter: int,
    n_batch: int,
    n_in: int,
    n_rec: int,
    learning_signal_mode: str,
    n_out: int = 3,
    lr: float = 5e-3,
    c_reg: float = 150.0,
    f_target: float = 20.0,
    tau_a_ms: float = 2000.0,
    gamma: float = 0.3,
    threshold: float = 0.03,
    w_gain: float = 1.0,
    trayectorias=None,
):
    print("At the start of run_training:")
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}  |  n_in={n_in}  n_rec={n_rec}  "
          f"n_iter={n_iter}  n_batch={n_batch}")

    n_samples = len(targets_np)

    model = HandwritingSNN(
        n_in=n_in, n_rec=n_rec, n_out=n_out,
        c_reg=c_reg, f_target=f_target,
        learning_signal_mode=learning_signal_mode,
        tau_a_ms=tau_a_ms,
        gamma=gamma,
        threshold=threshold,
        w_gain=w_gain,
    ).to(device)

    # Adam with same hyper-parameters as NEST (eta, beta_1, beta_2, epsilon)
    optimizer = optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,
    )

    spikes_torch = {
        k: torch.from_numpy(v).to(device)
        for k, v in spikes_dict.items()
    }

    loss_history = []

    targets_torch = torch.from_numpy(targets_np).to(device)

    for it in tqdm(range(n_iter), desc="E-prop"):

        idx = np.random.choice(n_samples, n_batch, replace=True)

        x_b = torch.stack(
            [spikes_torch[j] for j in idx],
            dim=0
        )
        t_b = targets_torch[idx]  # (n_batch, seq_T, 3)

        optimizer.zero_grad(set_to_none=True)
        out = model(x_b, targets=t_b, log_step=it)   # e-prop gradients assigned inside forward()

        # MSE (for logging only — no .backward() needed)
        loss_val = 0.5 * float((out - t_b).pow(2).sum(dim=-1).mean())
        loss_history.append(loss_val)

        optimizer.step()

        wandb.log({"train/loss": loss_val})

        if (it + 1) % 10 == 0 and trayectorias is not None:
            _log_character_images_to_wandb(
                model, spikes_torch, letras_ids, targets_torch, trayectorias, it + 1
            )

        if (it + 1) % 50 == 0:
            log_gradient_health(model, x_b, t_b, step=it + 1)

        if (it + 1) % 100 == 0:
            print(f"  iter {it + 1:5d}/{n_iter}  loss={loss_val:.6f}")

    # Full inference pass to collect all predictions
    model.eval()
    all_out = []

    with torch.no_grad():
        for i in range(0, n_samples, n_batch):

            end = min(i + n_batch, n_samples)

            idx = range(i, end)

            x_b = torch.stack(
                [spikes_torch[j] for j in idx],
                dim=0
            )

            all_out.append(model(x_b).cpu().numpy())

    outputs_np = np.concatenate(all_out, axis=0)

    return model, outputs_np, loss_history


# ── sweep-compatible training pipeline ────────────────────────────────────────

def build_and_train(dataset_path: str, output_dir: str, n_iter: int, n_batch: int):
    """
    Core pipeline. Reads all sweep hyperparams from wandb.config.
    wandb.init() must be called before this function.
    """
    cfg = wandb.config

    # fixed (non-swept) hyperparams
    n_in       = 200
    n_out      = 3
    f_target   = 20.0
    data_point = 8

    data, authors, symbols = load_dataset2(dataset_path)
    n_authors = len(authors)
    n_letras  = len(symbols)
    seq_T = len(data[0][0][0]) * data_point

    # Flatten (author, symbol, instance) triples into an ordered list so
    # both datos_letras (for build_targets) and spikes_dict share the same
    # integer index i for each (ai, si, inst_idx) triple.
    flat_keys    = []
    datos_letras = []
    for ai in range(n_authors):
        for si in range(n_letras):
            for inst_idx, arr in enumerate(data[ai].get(si, [])):
                flat_keys.append((ai, si, inst_idx))
                datos_letras.append(arr)

    n_sequences = len(datos_letras)
    traj_idx_per_seq = np.arange(n_sequences, dtype=int)

    # Log fixed hyperparams alongside the sweep params
    wandb.config.update({
        "n_in":        n_in,
        "n_out":       n_out,
        "n_authors":   n_authors,
        "n_letras":    n_letras,
        "n_sequences": n_sequences,
        "n_iter":      n_iter,
        "n_batch":     n_batch,
        "f_target_hz": f_target,
        "data_point":  data_point,
        "seq_T":       seq_T,
        "dataset_path": dataset_path,
        "rng_seed":    rng_seed,
        "tau_m_ms":    30.0,
        "tau_out_ms":  50.0,
    }, allow_val_change=True)

    targets_np = build_targets(datos_letras, n_sequences, seq_T, data_point, traj_idx_per_seq)

    print("Generating spikes...")
    spikes_keyed = generate_spikes_character_and_style(
        n_in=n_in,
        seq_T=seq_T,
        n_letras=n_letras,
        n_authors=n_authors,
        data=data,
        prob=float(cfg.prob),
    )
    spikes_dict = {i: spikes_keyed[k] for i, k in enumerate(flat_keys)}

    trayectorias = [
        f"{authors[ai]}_{symbols[si]}_{inst_idx}"
        for ai, si, inst_idx in flat_keys
    ]

    os.makedirs(output_dir, exist_ok=True)

    print("Starting training...")
    model, outputs_np, loss_history = run_training(
        spikes_dict, targets_np, traj_idx_per_seq, n_iter, n_batch,
        n_in, int(cfg.n_rec),
        learning_signal_mode=cfg.learning_signal_mode,
        n_out=n_out,
        lr=float(cfg.lr),
        c_reg=float(cfg.c_reg),
        f_target=f_target,
        tau_a_ms=float(cfg.tau_a_ms),
        gamma=float(cfg.gamma),
        threshold=float(cfg.threshold),
        w_gain=float(cfg.w_gain),
        trayectorias=trayectorias,
    )

    analyze_and_plot(
        outputs_np, targets_np, traj_idx_per_seq, trayectorias,
        output_dir, loss_history,
    )

    with open(f"{output_dir}/modelo_char_style.pkl", "wb") as f:
        pickle.dump({
            "state_dict": model.state_dict(),
            "n_in": n_in, "n_rec": int(cfg.n_rec), "n_out": n_out,
        }, f)
    print(f"Model saved to {output_dir}/modelo_char_style.pkl")


def main():

    parser = argparse.ArgumentParser(description="PyTorch e-prop handwriting generation")

    parser.add_argument(
        "--dataset-path", "-d", type=str, default=None,
        help="Path to the dataset directory (default depends on modality)",
    )
    parser.add_argument(
        "--learning_signal",
        type=str,
        default="symmetric",
        choices=["symmetric", "random", "adaptive"],
        help="Learning signal mode for standalone runs (ignored during sweeps)",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="W&B sweep ID — run this agent as part of a sweep (create with: wandb sweep sweep.yaml)",
    )
    parser.add_argument(
        "--sweep-count",
        type=int,
        default=None,
        help="Number of runs this agent will execute (default: unlimited)",
    )
    args = parser.parse_args()

    print("At the start of main():")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())

    n_iter     = 1200
    n_batch    = 32
    output_dir = "char_style_outputs_pytorch"

    dataset_path = args.dataset_path or "/data/gasbert/TFM_SNN/HOMUS_PROCESSED_mini"
    #dataset_path = args.dataset_path or "/data/113-2/users/gasbert/HOMUS_PROCESSED_mini"

    if args.sweep_id:
        # ── sweep agent mode ──────────────────────────────────────────────────
        # wandb.agent calls _sweep_run() for each trial; the sweep controller
        # fills wandb.config with the sampled hyperparams before the call.
        def _sweep_run():
            wandb.init()
            build_and_train(dataset_path, output_dir, n_iter, n_batch)
            wandb.finish()

        wandb.agent(args.sweep_id, _sweep_run, count=args.sweep_count)

    else:
        # ── standalone run ────────────────────────────────────────────────────
        config = {**DEFAULT_CONFIG, "learning_signal_mode": args.learning_signal}
        wandb.init(project="snn-handwriting", config=config)
        build_and_train(dataset_path, output_dir, n_iter, n_batch)
        wandb.finish()


if __name__ == "__main__":
    main()
