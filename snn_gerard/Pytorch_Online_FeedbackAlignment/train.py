"""
train.py — PyTorch e-prop handwriting generation.

Mirrors the pipeline of nest_handwriting_eprop_with_pen.py but uses PyTorch.

Modalities
----------
1. Style variation  – base + jitter-style spike encoding, multiple styles per letter
3. Alphabet generation – one frozen spike pattern per letter
"""

import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pickle
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import wandb

from models import HandwritingSNN


# ── reproducibility ────────────────────────────────────────────────────────────
rng_seed = 27
np.random.seed(rng_seed)
torch.manual_seed(rng_seed)


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
        targets[i, :, 0] = np.interp(x_eval, x_data, dx_c).astype(np.float32)
        targets[i, :, 1] = np.interp(x_eval, x_data, dy_c).astype(np.float32)
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
        wandb.log({f"images/{nombre_base}": wandb.Image(fig)})
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


# ── core training loop ─────────────────────────────────────────────────────────

def run_training(
    spikes_dict,
    letras_por_secuencia,   # (n_sequences,) int array: index into spikes_dict
    n_sequences,     
    targets_np,     # (n_sequences, seq_T, 3)
    n_iter: int,
    n_batch: int,
    n_in: int,
    n_rec: int,
    n_out: int = 3,
    lr: float = 5e-3,
    c_reg: float = 150.0,
    f_target: float = 20.0,
):
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}  |  n_in={n_in}  n_rec={n_rec}  "
          f"n_iter={n_iter}  n_batch={n_batch}")

    model = HandwritingSNN(
        n_in=n_in, n_rec=n_rec, n_out=n_out,
        c_reg=c_reg, f_target=f_target,
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
        start = (it * n_batch) % n_sequences
        idx = np.arange(start, start + n_batch) % n_sequences

        x_b = torch.stack(
            [spikes_torch[int(letras_por_secuencia[j])] for j in idx],
            dim=0
        )
        t_b = targets_torch[idx]  # (n_batch, seq_T, 3)

        optimizer.zero_grad(set_to_none=True)
        out = model(x_b, targets=t_b)   # e-prop gradients assigned inside forward()

        # MSE (for logging only — no .backward() needed)
        loss_val = 0.5 * float((out - t_b).pow(2).sum(dim=-1).mean())
        loss_history.append(loss_val)

        optimizer.step()

        wandb.log({"train/loss": loss_val}, step=it + 1)

        if (it + 1) % 100 == 0:
            print(f"  iter {it + 1:5d}/{n_iter}  loss={loss_val:.6f}")

    # Full inference pass to collect all predictions
    model.eval()
    all_out = []

    with torch.no_grad():
        for i in range(0, n_sequences, n_batch):

            end = min(i + n_batch, n_sequences)

            idx = range(i, end) 

            x_b = torch.stack(
                [spikes_torch[int(letras_por_secuencia[j])] for j in idx],
                dim=0
            )

            all_out.append(model(x_b).cpu().numpy())

    outputs_np = np.concatenate(all_out, axis=0)

    return model, outputs_np, loss_history


def main():
    print("Choose a modality:")
    print("  1. Style variation   (base + jitter encoding)")
    print("  3. Alphabet generation (one spike pattern per letter)")
    tipo_modalidad = int(input("Enter choice [1/3]: ").strip())

    # ── Modality 1: Style Variation ────────────────────────────────────────────
    if tipo_modalidad == 1:
        n_letras_distintas = int(input("Number of distinct characters: ").strip())
        n_estilos_por_letra = int(input("Number of styles per character: ").strip())
        n_letras = n_letras_distintas * n_estilos_por_letra

        n_iter  = 1000
        n_batch = n_letras_distintas
        n_base  = 280
        n_style = 20
        n_in    = n_base + n_style
        n_rec   = 500
        n_out   = 3
        lr      = 5e-3
        c_reg   = 150.0
        f_target = 20.0
        data_point = 8
        output_dir = "output_estilos_pytorch"

        dataset_path = input(
            "Dataset path [Dataset_estilos_A/Dataset_AA/prueba]: "
        ).strip() or "Dataset_estilos_A/Dataset_AA/prueba"

        datos_letras, trayectorias = load_dataset(dataset_path, n_letras)
        seq_T = len(datos_letras[0]) * data_point
        n_sequences = n_iter * n_batch

        wandb.init(
            project="snn-handwriting",
            config={
                "modality": 1,
                "n_letras_distintas": n_letras_distintas,
                "n_estilos_por_letra": n_estilos_por_letra,
                "n_letras": n_letras,
                "n_base": n_base,
                "n_style": n_style,
                "n_in": n_in,
                "n_rec": n_rec,
                "n_out": n_out,
                "n_iter": n_iter,
                "n_batch": n_batch,
                "lr": lr,
                "c_reg": c_reg,
                "f_target_hz": f_target,
                "data_point": data_point,
                "seq_T": seq_T,
                "dataset_path": dataset_path,
                "rng_seed": rng_seed,
                # SNN architecture
                "tau_m_ms": 30.0,
                "tau_a_ms": 2000.0,
                "tau_out_ms": 50.0,
                "threshold": 0.03,
                "gamma": 0.3,
            },
        )

        # Build (letra, estilo) plan
        combinaciones = [
            (l, s)
            for l in range(n_letras_distintas)
            for s in range(n_estilos_por_letra)
        ]
        rep = math.ceil(n_sequences / len(combinaciones))
        letras_estilos = np.array((combinaciones * rep)[:n_sequences])   # (n_sequences, 2)

        # Each sequence maps to a unique trajectory file
        traj_idx_per_seq = np.array(
            [(letras_estilos[i, 0] * n_estilos_por_letra + letras_estilos[i, 1]) % n_letras
             for i in range(n_sequences)],
            dtype=int,
        )

        targets_np = build_targets(datos_letras, n_sequences, seq_T, data_point, traj_idx_per_seq)
        spikes_np  = generate_spikes_modal1(
            n_base, n_style, n_letras_distintas, n_estilos_por_letra,
            n_sequences, seq_T, letras_estilos,
        )

        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/letras_estilos_por_secuencia.npy", letras_estilos)

        model, outputs_np, loss_history = run_training(
            spikes_np, targets_np, n_iter, n_batch, n_in, n_rec, n_out,
            lr=lr, c_reg=c_reg, f_target=f_target,
        )

        analyze_and_plot(
            outputs_np, targets_np, traj_idx_per_seq, trayectorias,
            output_dir, loss_history,
        )

        with open(f"{output_dir}/modelo.pkl", "wb") as f:
            pickle.dump({
                "state_dict": model.state_dict(),
                "n_in": n_in, "n_rec": n_rec, "n_out": n_out,
            }, f)
        print(f"Model saved to {output_dir}/modelo.pkl")
        wandb.finish()

    # ── Modality 3: Alphabet Generation ───────────────────────────────────────
    elif tipo_modalidad == 3:
        n_letras_input = input("Number of letters to train [1]: ").strip()
        n_letras   = int(n_letras_input) if n_letras_input else 1
        n_iter     = 1000
        n_batch    = 32
        n_in       = 200
        n_rec      = 400
        n_out      = 3
        lr         = 5e-3
        c_reg      = 150.0
        f_target   = 20.0
        data_point = 8
        output_dir = "abecedario_outputs_pytorch"

        default_path = "/home-local/gasbert/TFM_SNN_Offline_Generation/snn_marc/input_characters"
        path_input = input(f"Dataset path [{default_path}]: ").strip()
        dataset_path = path_input if path_input else default_path

        datos_letras, trayectorias = load_dataset(dataset_path, n_letras)
        seq_T = len(datos_letras[0]) * data_point

        wandb.init(
            project="snn-handwriting",
            config={
                "modality": 3,
                "n_letras": n_letras,
                "n_in": n_in,
                "n_rec": n_rec,
                "n_out": n_out,
                "n_iter": n_iter,
                "n_batch": n_batch,
                "lr": lr,
                "c_reg": c_reg,
                "f_target_hz": f_target,
                "data_point": data_point,
                "seq_T": seq_T,
                "dataset_path": dataset_path,
                "rng_seed": rng_seed,
                # SNN architecture
                "tau_m_ms": 30.0,
                "tau_a_ms": 2000.0,
                "tau_out_ms": 50.0,
                "threshold": 0.03,
                "gamma": 0.3,
            },
        )

        letras_por_secuencia = np.random.randint(0, n_letras, size=256)
        n_sequences = len(letras_por_secuencia)

        targets_np = build_targets(datos_letras, n_sequences, seq_T, data_point, letras_por_secuencia)
        print("Generating spikes...")
        spikes_dict = generate_spikes_modal3(
            n_in=n_in,
            seq_T=seq_T,
            n_letras=n_letras,
        )
        print("Saving sequences-to-character mapping...")
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/letras_por_secuencia.npy", letras_por_secuencia)

        print("Starting training...")
        model, outputs_np, loss_history = run_training(
            spikes_dict, letras_por_secuencia, n_sequences, targets_np, n_iter, n_batch, n_in, n_rec, n_out,
            lr=lr, c_reg=c_reg, f_target=f_target,
        )

        analyze_and_plot(
            outputs_np, targets_np, letras_por_secuencia, trayectorias,
            output_dir, loss_history,
        )

        with open(f"{output_dir}/modelo_abecedario.pkl", "wb") as f:
            pickle.dump({
                "state_dict": model.state_dict(),
                "n_in": n_in, "n_rec": n_rec, "n_out": n_out,
            }, f)
        print(f"Model saved to {output_dir}/modelo_abecedario.pkl")
        wandb.finish()

    else:
        print("Modality not implemented.")


if __name__ == "__main__":
    main()
