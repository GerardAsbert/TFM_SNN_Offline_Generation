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
from PIL import Image

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import wandb

from models import HandwritingSNN, render_trajectory
from grad_health import log_gradient_health


# ── reproducibility ────────────────────────────────────────────────────────────
rng_seed = 27
np.random.seed(rng_seed)
torch.manual_seed(rng_seed)


# ── sweep defaults ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "threshold":            1.0,
    "w_gain":               1.0,
    "lr":                   1e-3,
    "gamma":                0.3,
    "c_reg":                1.0,
    "n_rec":                200,
    "tau_a_ms":             2000,
    "prob":                 0.05,
    "learning_signal_mode": "random",
}


# ── dataset helpers ────────────────────────────────────────────────────────────

def load_image_dataset(dataset_path: str, img_H: int, img_W: int):
    """
    Load dataset with structure:
        dataset_path/<author>/<symbol>/<instance>.(png|jpg|...)
    Each image is padded to square, resized to (img_H, img_W), normalized to [0,1].

    Returns:
        data    : dict {author_idx: {symbol_idx: [np.ndarray(H,W), ...]}}
        authors : sorted list of author directory names
        symbols : sorted list of symbol directory names
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm")
    authors = sorted(
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    )
    if not authors:
        raise ValueError(f"No author directories found in {dataset_path}")

    all_symbols = set()
    for author in authors:
        ap = os.path.join(dataset_path, author)
        all_symbols.update(
            d for d in os.listdir(ap) if os.path.isdir(os.path.join(ap, d))
        )
    symbols = sorted(all_symbols)

    data = {}
    for ai, author in enumerate(authors):
        data[ai] = {}
        ap = os.path.join(dataset_path, author)
        for si, symbol in enumerate(symbols):
            sp = os.path.join(ap, symbol)
            if not os.path.isdir(sp):
                data[ai][si] = []
                continue
            files = sorted(f for f in os.listdir(sp) if f.lower().endswith(exts))
            data[ai][si] = [load_gt_image(os.path.join(sp, f), img_H, img_W) for f in files]
    return data, authors, symbols



def load_gt_image(path, H, W):
    im = Image.open(path).convert("L")
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("L", (s, s), 0)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    canvas = canvas.resize((W, H), Image.BILINEAR)
    return np.asarray(canvas, dtype=np.float32) / 255.0


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
    outputs,                # (n_sequences, seq_T, 3) predicted trajectories
    target_imgs,            # (n_sequences, H, W) ground-truth images
    traj_idx_per_seq,
    trayectorias,
    output_dir: str,
    loss_history=None,
    render_sigma: float = 1.5,
    img_H: int = 32,
    img_W: int = 32,
):
    """Post-training analysis: loss curve + per-symbol GT-vs-generated images."""
    os.makedirs(output_dir, exist_ok=True)

    if loss_history:
        fig_loss = plt.figure()
        plt.plot(range(1, len(loss_history) + 1), loss_history)
        plt.xlabel("training iteration"); plt.ylabel("image MSE loss")
        plt.title("Training loss"); plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_training.png", dpi=300)
        wandb.log({"charts/loss_curve": wandb.Image(fig_loss)})
        plt.close()

    # render all predicted trajectories to images (chunked to bound memory)
    gen_list = []
    with torch.no_grad():
        outs_t = torch.from_numpy(outputs.astype(np.float32))
        for i in range(0, outs_t.shape[0], 64):
            gen_list.append(
                render_trajectory(outs_t[i:i + 64], img_H, img_W, render_sigma).cpu().numpy()
            )
    gen = np.concatenate(gen_list, axis=0)          # (n_sequences, H, W)

    mse_all = float(np.mean((gen - target_imgs) ** 2))
    print(f"Image MSE (all): {mse_all:.6f}")
    wandb.log({"val/img_mse": mse_all})

    for li in sorted(np.unique(traj_idx_per_seq)):
        nombre_base = os.path.splitext(trayectorias[li])[0]
        last_i = np.where(traj_idx_per_seq == li)[0][-1]
        g, t = gen[last_i], target_imgs[last_i]
        mse = float(np.mean((g - t) ** 2))

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(t, cmap="gray", vmin=0, vmax=1); axes[0].set_title("target"); axes[0].axis("off")
        axes[1].imshow(g, cmap="gray", vmin=0, vmax=1); axes[1].set_title("generated"); axes[1].axis("off")
        fig.suptitle(f"{nombre_base}  MSE:{mse:.4f}"); plt.tight_layout()
        plt.savefig(f"{output_dir}/{nombre_base}.png", dpi=300); plt.close()
        plt.imsave(f"{output_dir}/{nombre_base}_gen.png", g, cmap="gray", vmin=0, vmax=1)


# ── periodic image logging helper ─────────────────────────────────────────────

def _log_character_images_to_wandb(
    model, spikes_torch, letras_por_secuencia, target_imgs_torch, trayectorias, step,
    render_sigma=1.5, img_H=32, img_W=32,
):
    """Run inference and log GT-vs-generated images per unique symbol to wandb."""
    letras_unicas = sorted(set(int(l) for l in letras_por_secuencia))
    with torch.no_grad():
        for li in letras_unicas:
            nombre_base = os.path.splitext(trayectorias[li])[0]
            last_i = np.where(np.array(letras_por_secuencia) == li)[0][-1]

            x = spikes_torch[li].unsqueeze(0)                 # (1, T, n_in)
            out = model(x)                                    # (1, T, 3)
            gen = render_trajectory(out, img_H, img_W, render_sigma)[0].cpu().numpy()
            tgt = target_imgs_torch[last_i].cpu().numpy()
            mse = float(np.mean((gen - tgt) ** 2))

            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(tgt, cmap="gray", vmin=0, vmax=1); axes[0].set_title("target"); axes[0].axis("off")
            axes[1].imshow(gen, cmap="gray", vmin=0, vmax=1); axes[1].set_title("generated"); axes[1].axis("off")
            fig.suptitle(f"{nombre_base}  iter {step}  MSE:{mse:.4f}"); plt.tight_layout()
            wandb.log({f"images/{nombre_base}": wandb.Image(fig)})
            plt.close()


# ── core training loop ─────────────────────────────────────────────────────────

def run_training(
    spikes_dict,
    target_imgs_np,  # (n_sequences, H, W)
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
    render_sigma: float = 1.5,
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

    n_samples = len(target_imgs_np)
    img_H, img_W = target_imgs_np.shape[1], target_imgs_np.shape[2]

    model = HandwritingSNN(
        n_in=n_in, n_rec=n_rec, n_out=n_out,
        c_reg=c_reg, f_target=f_target,
        learning_signal_mode=learning_signal_mode,
        tau_a_ms=tau_a_ms,
        gamma=gamma,
        threshold=threshold,
        w_gain=w_gain,
        img_H=img_H, img_W=img_W, render_sigma=render_sigma,
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

    target_imgs_torch = torch.from_numpy(target_imgs_np).to(device)

    for it in tqdm(range(n_iter), desc="E-prop"):

        idx = np.random.choice(n_samples, n_batch, replace=True)

        x_b = torch.stack(
            [spikes_torch[j] for j in idx],
            dim=0
        )
        img_b = target_imgs_torch[idx]   # (n_batch, H, W)

        optimizer.zero_grad(set_to_none=True)
        out = model(x_b, target_image=img_b, log_step=it)   # e-prop grads assigned inside forward()

        # image-space MSE (logging only)
        with torch.no_grad():
            pred_img = render_trajectory(out, img_H, img_W, render_sigma)
            loss_val = 0.5 * float((pred_img - img_b).pow(2).mean())

            # ── pen-up collapse diagnostics ──────────────────────────────────
            pen_prob = torch.sigmoid(out[:, :, 2] * 5.0)        # (B, T), same as renderer
            wandb.log({
                "pendiag/pen_prob_mean":   pen_prob.mean().item(),     # → 0 means collapse
                "pendiag/pen_logit_mean":  out[:, :, 2].mean().item(), # raw pen channel
                "pendiag/pred_img_mean":   pred_img.mean().item(),     # should track target
                "pendiag/target_img_mean": img_b.mean().item(),        # reference (constant-ish)
                "pendiag/pred_ink_frac":   (pred_img > 0.1).float().mean().item(),
                "pendiag/target_ink_frac": (img_b   > 0.1).float().mean().item(),
            })
        loss_history.append(loss_val)

        optimizer.step()

        wandb.log({"train/loss": loss_val})

        if (it + 1) % 10 == 0 and trayectorias is not None:
            _log_character_images_to_wandb(
                model, spikes_torch, letras_ids, target_imgs_torch, trayectorias,
                it + 1, render_sigma, img_H, img_W,
            )

        if (it + 1) % 50 == 0:
            log_gradient_health(model, x_b, img_b, step=it + 1)

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

    IMG_H, IMG_W = 32, 32          # fixed canvas; every GT image normalized to this
    seq_T = 256                     # number of SNN timesteps (free hyperparameter now)

    data, authors, symbols = load_image_dataset(dataset_path, IMG_H, IMG_W)
    n_authors = len(authors)
    n_letras  = len(symbols)

    # Flatten (author, symbol, instance) triples so target_imgs and spikes_dict
    # share the same integer index for each triple.
    flat_keys   = []
    target_imgs = []
    for ai in range(n_authors):
        for si in range(n_letras):
            for inst_idx, img in enumerate(data[ai].get(si, [])):
                flat_keys.append((ai, si, inst_idx))
                target_imgs.append(img)

    n_sequences = len(target_imgs)
    traj_idx_per_seq = np.arange(n_sequences, dtype=int)
    target_imgs_np = np.stack(target_imgs, axis=0).astype(np.float32)  # (n_seq, H, W)

    wandb.config.update({
        "n_in": n_in, "n_out": n_out,
        "n_authors": n_authors, "n_letras": n_letras, "n_sequences": n_sequences,
        "n_iter": n_iter, "n_batch": n_batch,
        "f_target_hz": f_target, "seq_T": seq_T,
        "img_H": IMG_H, "img_W": IMG_W,
        "dataset_path": dataset_path, "rng_seed": rng_seed,
        "tau_m_ms": 30.0, "tau_out_ms": 50.0,
    }, allow_val_change=True)

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
        spikes_dict, target_imgs_np, traj_idx_per_seq, n_iter, n_batch,
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
        outputs_np, target_imgs_np, traj_idx_per_seq, trayectorias,
        output_dir, loss_history,
        render_sigma=1.5, img_H=IMG_H, img_W=IMG_W,
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

    dataset_path = args.dataset_path or "/data/gasbert/TFM_SNN/FORNES_mini_offline"
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
