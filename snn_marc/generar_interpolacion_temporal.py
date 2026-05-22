import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nest
import pickle

# ===================== CARGA MODELO ENTRENADO =====================
with open("interpolacion/interpolacion.pkl", "rb") as f:
    modelo = pickle.load(f)

pesos = {
    "in_rec": modelo["pesos_in_rec"],
    "rec_rec": modelo["pesos_rec_rec"],
    "rec_out": modelo["pesos_rec_out"],
}
n_in     = modelo["n_in"]
n_rec    = modelo["n_rec"]
n_out    = modelo["n_out"]
duration = modelo["duration"]
steps    = modelo["steps"]
steps_seq = steps["sequence"]

# Deben coincidir con el entrenamiento
n_base  = 280
n_style = 20
assert n_base + n_style == n_in, f"n_base({n_base}) + n_style({n_style}) != n_in({n_in})"

# ===================== CARGAR BASE Y ESTILOS EXACTOS DEL ENTRENAMIENTO =====================
spike_base      = np.load("output_estilos/base_letter0.npy").astype(bool)   # (n_base, T)
spike_style_A1  = np.load("output_estilos/style_A1.npy").astype(bool)       # (n_style, T)
spike_style_A2  = np.load("output_estilos/style_A2.npy").astype(bool)       # (n_style, T)

# Checks
T = steps_seq
assert spike_base.shape == (n_base, T),  f"Base shape {spike_base.shape} != ({n_base},{T})"
assert spike_style_A1.shape == (n_style, T)
assert spike_style_A2.shape == (n_style, T)

# ===================== HELPERS =====================
def aplicar_pesos(pesos_matrix, pre_neurons, post_neurons):
    """Aplica una matriz de pesos (post x pre) a la conectividad actual (all_to_all)."""
    conns = nest.GetConnections(pre_neurons, post_neurons)
    sources = np.array(nest.GetStatus(conns, "source"))
    targets = np.array(nest.GetStatus(conns, "target"))
    src_base = int(pre_neurons.tolist()[0])
    tgt_base = int(post_neurons.tolist()[0])
    weights = [pesos_matrix[tgt - tgt_base, src - src_base] for tgt, src in zip(targets, sources)]
    nest.SetStatus(conns, [{"weight": float(w)} for w in weights])

def interpolate_style_spikes(style_A1, style_A2, alpha):
    """
    Interpolación por orden de spikes por neurona:
    - Empareja la k-ésima spike de A1 con la k-ésima de A2 (k=0..max-1).
    - new_t = round((1-alpha)*t1 + alpha*t2)
    - Si una serie tiene menos spikes, se extiende con el último índice.
    - Se recorta y se eliminan t=0.
    """
    n_neu, T = style_A1.shape
    out = np.zeros_like(style_A1, dtype=bool)
    for i in range(n_neu):
        idx1 = np.flatnonzero(style_A1[i])
        idx2 = np.flatnonzero(style_A2[i])

        if idx1.size == 0 and idx2.size == 0:
            continue

        m = max(idx1.size, idx2.size)
        if idx1.size == 0:
            t1 = np.zeros(m, dtype=int)
        else:
            t1 = np.pad(idx1, (0, m - idx1.size), mode='edge')
        if idx2.size == 0:
            t2 = np.zeros(m, dtype=int)
        else:
            t2 = np.pad(idx2, (0, m - idx2.size), mode='edge')

        new_idx = np.round((1.0 - alpha) * t1 + alpha * t2).astype(int)
        new_idx = np.clip(new_idx, 0, T - 1)
        out[i, np.unique(new_idx)] = True  # evita duplicados en misma columna

    out[:, 0] = False  # nunca dejamos spikes en t=0
    return out

def construir_full_input(spike_base, spike_style):
    """Concatena base (arriba) + estilo (debajo) -> (n_in, T)."""
    return np.vstack([spike_base, spike_style])

def simular_y_guardar(spike_train_full, nombre_png, out_dir):
    """
    Simula la red con el spike_train_full (n_in, T) y guarda la trayectoria (readout x,y).
    """
    T = spike_train_full.shape[1]

    nest.ResetKernel()
    nest.SetKernelStatus({
        "resolution": duration["step"],
        "rng_seed": 1,
        "local_num_threads": 1
    })

    # Neuronas y conexiones
    gen_spk_in = nest.Create("spike_generator", n_in)
    nrns_in = nest.Create("parrot_neuron", n_in)

    tau_m_mean = 30.0
    params_nrn_rec = {
        "adapt_tau": 2000.0,
        "C_m": 250.0,
        "c_reg": 150.0,
        "E_L": 0.0,
        "f_target": 20.0,
        "gamma": 0.3,
        "I_e": 0.0,
        "regular_spike_arrival": False,
        "surrogate_gradient_function": "piecewise_linear",
        "t_ref": 0.0,
        "tau_m": nest.random.normal(mean=tau_m_mean, std=2.0),
        "V_m": 0.0,
        "V_th": 0.03,
    }
    params_nrn_rec["adapt_beta"] = (
        1.7 * (1.0 - np.exp(-1 / params_nrn_rec["adapt_tau"])) / (1.0 - np.exp(-1.0 / tau_m_mean))
    )
    params_nrn_out = {
        "C_m": 1.0,
        "E_L": 0.0,
        "I_e": 0.0,
        "loss": "mean_squared_error",
        "regular_spike_arrival": False,
        "tau_m": 50.0,
        "V_m": 0.0,
    }

    nrns_rec = nest.Create("eprop_iaf_adapt_bsshslm_2020", n_rec, params_nrn_rec)
    nrns_out = nest.Create("eprop_readout_bsshslm_2020", n_out, params_nrn_out)

    nest.Connect(gen_spk_in, nrns_in, "one_to_one")
    nest.Connect(nrns_in, nrns_rec, "all_to_all")
    nest.Connect(nrns_rec, nrns_rec, "all_to_all")
    nest.Connect(nrns_rec, nrns_out, "all_to_all")

    # Cargar spike times
    spike_times = np.arange(1, T + 1)
    nest.SetStatus(gen_spk_in, [
        {"spike_times": spike_times[spike_train_full[i]].astype(np.float32)} for i in range(n_in)
    ])

    # Aplicar pesos entrenados
    aplicar_pesos(pesos["in_rec"], nrns_in, nrns_rec)
    aplicar_pesos(pesos["rec_rec"], nrns_rec, nrns_rec)
    aplicar_pesos(pesos["rec_out"], nrns_rec, nrns_out)

    # Registrar salida
    mm = nest.Create("multimeter", params={
        "record_from": ["readout_signal"],
        "interval": 1.0,
        "start": 0.0,
        "stop": float(T),
    })
    nest.Connect(mm, nrns_out)

    nest.Simulate(float(T))

    events = mm.get("events")
    senders = events["senders"]
    signal = events["readout_signal"]
    out_x = signal[senders == np.min(senders)]
    out_y = -signal[senders == np.max(senders)]

    plt.figure(figsize=(4, 3))
    plt.plot(out_x, out_y)
    plt.title(nombre_png)
    plt.axis("equal")
    plt.xticks([]); plt.yticks([]); plt.tight_layout()
    path_fig = os.path.join(out_dir, f"{nombre_png}.png")
    plt.savefig(path_fig, dpi=300)
    plt.close()

# ===================== INTERPOLACIÓN (0%..100% EN 10%) =====================
output_dir = "interpolacion_temporal"
imgs_dir = os.path.join(output_dir, "imgs")
os.makedirs(imgs_dir, exist_ok=True)

# 0% y 100% de referencia
full_A1 = construir_full_input(spike_base, spike_style_A1)
full_A2 = construir_full_input(spike_base, spike_style_A2)
simular_y_guardar(full_A1, "interp_00", imgs_dir)
simular_y_guardar(full_A2, "interp_100", imgs_dir)

# 10%,20%,...,90%
for k in range(1, 10):
    alpha = k / 10.0
    style_interp = interpolate_style_spikes(spike_style_A1, spike_style_A2, alpha)
    full_interp = construir_full_input(spike_base, style_interp)
    nombre = f"interp_{int(alpha*100):02d}"
    simular_y_guardar(full_interp, nombre, imgs_dir)

# ===================== COLLAGE =====================
names = [f"interp_{p:02d}" for p in range(0, 101, 10)]
n_cols = 6
n_rows = int(np.ceil(len(names) / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.0))

for idx, nombre in enumerate(names):
    r, c = divmod(idx, n_cols)
    ax = axs[r, c] if n_rows > 1 else axs[c]
    path_img = os.path.join(imgs_dir, f"{nombre}.png")
    if os.path.exists(path_img):
        img = mpimg.imread(path_img)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(nombre, fontsize=8)
    else:
        ax.axis('off')
        ax.set_title(f"{nombre}\n(no encontrado)", fontsize=6)

# Oculta celdas sobrantes
for j in range(len(names), n_rows * n_cols):
    r, c = divmod(j, n_cols)
    ax = axs[r, c] if n_rows > 1 else axs[c]
    ax.axis("off")

plt.tight_layout()
collage_path = os.path.join(output_dir, "collage_interpolacion_0_100.png")
plt.savefig(collage_path, dpi=300)
plt.show()

print(f"✅ Imágenes individuales en: {imgs_dir}")
print(f"✅ Collage guardado en: {collage_path}")
