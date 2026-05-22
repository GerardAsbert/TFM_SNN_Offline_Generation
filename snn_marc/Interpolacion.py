import matplotlib as mpl
import matplotlib.pyplot as plt
import nest
import numpy as np
from cycler import cycler
from IPython.display import Image
import os
import pickle
from collections import defaultdict
from sklearn.metrics import mean_squared_error

try:
    Image(filename="./eprop_supervised_regression_schematic_handwriting.png")
except Exception:
    pass

# -------------------------
# 📌 PARÁMETROS ESCALABLES
# -------------------------
rng_seed = 1  # numpy random seed
np.random.seed(rng_seed)  # fix numpy random seed

frecuencia_cambio_letra = 1
n_letras = 2
n_batch = 2
n_iter = 10000
n_letras_distintas = 1
n_estilos_por_letra = 2
n_estilos = n_letras_distintas * n_estilos_por_letra
n_base = 280
n_style = 20
n_in = 300
n_rec = 500
n_out = 2  # number of readout neurons

steps = {
    "data_point": 6,
}

# -------------------------
# 📁 CARGAR LETRAS DESDE DATASET
# -------------------------
dataset_path = "Dataset_estilos_A/Dataset_AA"
trayectorias = sorted([f for f in os.listdir(dataset_path) if f.endswith(".txt")])

if n_letras > len(trayectorias):
    raise ValueError(f"Solo hay {len(trayectorias)} letras en la carpeta Dataset, pero pediste {n_letras}")

datos_letras = []
for i in range(n_letras):
    path_letra = os.path.join(dataset_path, trayectorias[i])
    datos = np.loadtxt(path_letra)
    datos_letras.append(datos)

# -------------------------
# ⚙️ CALCULAR STEPS Y DURACIONES
# -------------------------
steps["sequence"] = len(datos_letras[0]) * steps["data_point"]  # asumimos que todas tienen misma longitud
steps["learning_window"] = steps["sequence"]
steps["task"] = n_iter * n_batch * steps["sequence"]
steps.update({
    "offset_gen": 1,
    "delay_in_rec": 1,
    "delay_rec_out": 1,
    "delay_out_norm": 1,
    "extension_sim": 1,
})
steps["delays"] = steps["delay_in_rec"] + steps["delay_rec_out"] + steps["delay_out_norm"]
steps["total_offset"] = steps["offset_gen"] + steps["delays"]
steps["sim"] = steps["task"] + steps["total_offset"] + steps["extension_sim"]

duration = {"step": 1.0}
duration.update({key: value * duration["step"] for key, value in steps.items()})

params_setup = {
    "eprop_learning_window": duration["learning_window"],
    "eprop_reset_neurons_on_update": True,  # if True, reset dynamic variables at start of each update interval
    "eprop_update_interval": duration["sequence"],  # ms, time interval for updating the synaptic weights
    "print_time": True,  # if True, print time progress bar during simulation, set False if run as code cell
    "resolution": duration["step"],
    "local_num_threads": 16,  # number of virtual processes, set in case of distributed computing
    "rng_seed": rng_seed,  # seed for NEST random generator
}

nest.ResetKernel()
nest.set(**params_setup)

tau_m_mean = 30.0  # ms, mean of membrane time constant distribution

params_nrn_rec = {
    "adapt_tau": 2000.0,  # ms, time constant of adaptive threshold
    "C_m": 250.0,  # pF
    "c_reg": 150.0,  # firing rate regularization scaling
    "E_L": 0.0,  # mV
    "f_target": 20.0,  # spikes/s
    "gamma": 0.3,  # scaling of the pseudo derivative
    "I_e": 0.0,  # pA
    "regular_spike_arrival": False,
    "surrogate_gradient_function": "piecewise_linear",
    "t_ref": 0.0,  # ms
    "tau_m": nest.random.normal(mean=tau_m_mean, std=2.0),  # ms
    "V_m": 0.0,  # mV
    "V_th": 0.03,  # mV
}
params_nrn_rec["adapt_beta"] = (
    1.7 * (1.0 - np.exp(-1 / params_nrn_rec["adapt_tau"])) / (1.0 - np.exp(-1.0 / tau_m_mean))
)  # prefactor of adaptive threshold

params_nrn_out = {
    "C_m": 1.0,
    "E_L": 0.0,
    "I_e": 0.0,
    "loss": "mean_squared_error",  # loss function
    "regular_spike_arrival": False,
    "tau_m": 50.0,
    "V_m": 0.0,
}

gen_spk_in = nest.Create("spike_generator", n_in)
nrns_in = nest.Create("parrot_neuron", n_in)
nrns_rec = nest.Create("eprop_iaf_adapt_bsshslm_2020", n_rec, params_nrn_rec)
nrns_out = nest.Create("eprop_readout_bsshslm_2020", n_out, params_nrn_out)
gen_rate_target = nest.Create("step_rate_generator", n_out)

n_record = 1
n_record_w = 3
if n_record == 0 or n_record_w == 0:
    raise ValueError("n_record and n_record_w >= 1 required")

params_mm_out = {
    "interval": duration["step"],
    "record_from": ["V_m", "readout_signal", "readout_signal_unnorm", "target_signal", "error_signal"],
    "start": duration["total_offset"],
    "stop": duration["total_offset"] + duration["task"],
}
mm_out = nest.Create("multimeter", params_mm_out)

nrns_rec_record = nrns_rec[:n_record]

params_conn_all_to_all = {"rule": "all_to_all", "allow_autapses": False}
params_conn_one_to_one = {"rule": "one_to_one"}

dtype_weights = np.float32
weights_in_rec = np.array(np.random.randn(n_in, n_rec).T / np.sqrt(n_in), dtype=dtype_weights)
weights_rec_rec = np.array(np.random.randn(n_rec, n_rec).T / np.sqrt(n_rec), dtype=dtype_weights)
np.fill_diagonal(weights_rec_rec, 0.0)  # no autapses
weights_rec_out = np.array(np.random.randn(n_rec, n_out).T / np.sqrt(n_rec), dtype=dtype_weights)
weights_out_rec = np.array(np.random.randn(n_rec, n_out) / np.sqrt(n_rec), dtype=dtype_weights)

params_common_syn_eprop = {
    "optimizer": {
        "type": "adam",
        "batch_size": n_batch,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "eta": 5e-3,
        "Wmin": -100.0,
        "Wmax": 100.0,
    },
    "average_gradient": False,
}
params_syn_base = {
    "synapse_model": "eprop_synapse_bsshslm_2020",
    "delay": duration["step"],  # ms
    "tau_m_readout": params_nrn_out["tau_m"],
}
params_syn_in = params_syn_base.copy()
params_syn_in["weight"] = weights_in_rec
params_syn_rec = params_syn_base.copy()
params_syn_rec["weight"] = weights_rec_rec
params_syn_out = params_syn_base.copy()
params_syn_out["weight"] = weights_rec_out

params_syn_feedback = {
    "synapse_model": "eprop_learning_signal_connection_bsshslm_2020",
    "delay": duration["step"],
    "weight": weights_out_rec,
}
params_syn_rate_target = {
    "synapse_model": "rate_connection_delayed",
    "delay": duration["step"],
    "receptor_type": 2,
}
params_syn_static = {
    "synapse_model": "static_synapse",
    "delay": duration["step"],
}
params_init_optimizer = {
    "optimizer": {
        "m": 0.0,
        "v": 0.0,
    }
}

nest.SetDefaults("eprop_synapse_bsshslm_2020", params_common_syn_eprop)

nest.Connect(gen_spk_in, nrns_in, params_conn_one_to_one, params_syn_static)   # 1
nest.Connect(nrns_in, nrns_rec, params_conn_all_to_all, params_syn_in)         # 2
nest.Connect(nrns_rec, nrns_rec, params_conn_all_to_all, params_syn_rec)       # 3
nest.Connect(nrns_rec, nrns_out, params_conn_all_to_all, params_syn_out)       # 4
nest.Connect(nrns_out, nrns_rec, params_conn_all_to_all, params_syn_feedback)  # 5
nest.Connect(gen_rate_target, nrns_out, params_conn_one_to_one, params_syn_rate_target)  # 6
nest.Connect(mm_out, nrns_out, params_conn_all_to_all, params_syn_static)

# After creating the connections, initialize optimizer dyn vars for a couple of connections
conns = nest.GetConnections(nrns_rec[0], nrns_rec[1:3])
if len(conns) > 0:
    nest.SetStatus(conns, [params_init_optimizer] * len(conns))

# ====================== ENTRADAS (BASE + ESTILO CON JITTER) ======================
# Recalcular para asegurar consistencia si se cambia n_base/n_style
n_in = n_base + n_style
assert n_base + n_style == n_in, f"El total de neuronas ({n_base} + {n_style}) debe ser igual a n_in ({n_in})"

input_spike_prob = 0.05
n_sequences = n_iter * n_batch
dtype_in_spks = np.float32

def generar_spikes_entrenamiento():
    def jitter_spike_train(base_spikes, window=10):
        n_neu, T = base_spikes.shape
        out = np.zeros_like(base_spikes, dtype=bool)
        for i in range(n_neu):
            idx = np.flatnonzero(base_spikes[i])
            if idx.size == 0:
                continue
            shifts = np.random.randint(-window, window + 1, size=idx.size)
            new_idx = np.clip(idx + shifts, 0, T - 1)
            out[i, new_idx] = True
        out[:, 0] = False
        return out

    ventana_jitter = 10

    spikes_base_por_letra = []
    spikes_por_estilo = []

    # base por letra (n_base)
    for letra_idx in range(n_letras_distintas):
        np.random.seed(42 + letra_idx)
        spike_train_base = (np.random.rand(n_base, steps["sequence"]) < input_spike_prob)
        spike_train_base[0, :] = 0
        spikes_base_por_letra.append(spike_train_base)

    # estilos (n_style): estilo 0 aleatorio; estilo 1.. = jitter del 0
    for letra_idx in range(n_letras_distintas):
        seed_base = 100 + letra_idx
        np.random.seed(seed_base)
        estilo0 = (np.random.rand(n_style, steps["sequence"]) < input_spike_prob)
        estilo0[0, :] = 0
        spikes_por_estilo.append(estilo0)

        for estilo_idx in range(1, n_estilos_por_letra):
            seed_jitter = 200 + letra_idx + 1000 * estilo_idx
            np.random.seed(seed_jitter)
            estilo_derivado = jitter_spike_train(estilo0, window=ventana_jitter)
            estilo_derivado[0, :] = 0
            spikes_por_estilo.append(estilo_derivado)

    return spikes_base_por_letra, spikes_por_estilo

# Genera spikes de entrenamiento (reproducibles)
spikes_base_por_letra, spikes_por_estilo = generar_spikes_entrenamiento()

# Asignar combinaciones (letra_idx, estilo_idx) por secuencia
combinaciones = [(l, s) for l in range(n_letras_distintas) for s in range(n_estilos_por_letra)]
repeticiones = int(np.ceil(n_sequences / len(combinaciones)))
combinaciones_replicadas = (combinaciones * repeticiones)[:n_sequences]
letras_estilos_por_secuencia = np.array(combinaciones_replicadas)

# Guardar orden
os.makedirs("output_estilos", exist_ok=True)
np.save("output_estilos/letras_estilos_por_secuencia.npy", letras_estilos_por_secuencia)

# Construir booleano (n_in, n_sequences*T) con base+estilo
input_spike_bools_concat = []
for seq_idx in range(n_sequences):
    letra_idx, estilo_idx = letras_estilos_por_secuencia[seq_idx]
    estilo_global_idx = letra_idx * n_estilos_por_letra + estilo_idx
    spike_train_base = spikes_base_por_letra[letra_idx]
    estilo_spikes = spikes_por_estilo[estilo_global_idx]
    spikes_concat = np.vstack([spike_train_base, estilo_spikes])
    input_spike_bools_concat.append(spikes_concat)
input_spike_bools_concat = np.concatenate(input_spike_bools_concat, axis=1)
os.makedirs("output_estilos", exist_ok=True)

# Con n_letras_distintas = 1 y n_estilos_por_letra = 2:
#   - base de la letra 0
#   - estilo A1 = spikes_por_estilo[0]
#   - estilo A2 = spikes_por_estilo[1]  (jitter del A1)
np.save("output_estilos/base_letter0.npy",  spikes_base_por_letra[0].astype(np.uint8))
np.save("output_estilos/style_A1.npy",      spikes_por_estilo[0].astype(np.uint8))
np.save("output_estilos/style_A2.npy",      spikes_por_estilo[1].astype(np.uint8))

# Generar spike_times para spike_generator
spike_times_global = np.arange(duration["step"], duration["task"], duration["step"]) + duration["offset_gen"]
input_spike_bools_concat_no_t0 = input_spike_bools_concat[:, 1:]  # quitar t=0

params_gen_spk_in = []
for input_spike_bool in input_spike_bools_concat_no_t0:
    input_spike_times = spike_times_global[input_spike_bool]
    params_gen_spk_in.append({"spike_times": input_spike_times.astype(dtype_in_spks)})

nest.SetStatus(gen_spk_in, params_gen_spk_in)

# ====================== OBJETIVOS PARA gen_rate_target (dx, dy) ======================
x_eval = np.arange(steps["sequence"]) / steps["data_point"]
x_data = np.arange(steps["sequence"] // steps["data_point"])

# Normaliza/interpola las trayectorias del dataset
signal_letras = []
for data in datos_letras:  # cada .txt: (dx, dy) incrementales
    señales = []
    for y_data in np.cumsum(data, axis=0).T:  # pasar a coordenadas acumuladas
        y_data = y_data / np.max(np.abs(y_data))
        y_data = y_data - y_data[0]
        señales.append(np.interp(x_eval, x_data, y_data))
    signal_letras.append(señales)  # -> [ [dx_interp, dy_interp], ... ]

# Construye la señal completa concatenando secuencias en su orden
params_gen_rate_target = []
for comp in range(len(signal_letras[0])):  # 0=dx, 1=dy
    full_signal = []
    for i in range(n_sequences):
        letra_idx, estilo_idx = letras_estilos_por_secuencia[i]
        traj_idx = letra_idx * n_estilos_por_letra + estilo_idx  # 0 o 1
        full_signal.extend(signal_letras[traj_idx][comp])
    params_gen_rate_target.append({
        "amplitude_times": np.arange(0.0, duration["task"], duration["step"]) + duration["total_offset"],
        "amplitude_values": np.array(full_signal, dtype=np.float32),
    })

# Cargar objetivos en el generador (¡imprescindible!)
nest.SetStatus(gen_rate_target, params_gen_rate_target)

# 🔚 Spike final para forzar actualización
gen_spk_final_update = nest.Create("spike_generator", 1, {"spike_times": [duration["task"] + duration["delays"]]})
nest.Connect(gen_spk_final_update, nrns_in + nrns_rec, "all_to_all", {"weight": 1000.0})

def get_weights(pop_pre, pop_post):
    conns = nest.GetConnections(pop_pre, pop_post).get(["source", "target", "weight"])
    conns["senders"] = np.array(conns["source"]) - np.min(conns["source"])
    conns["targets"] = np.array(conns["target"]) - np.min(conns["target"])
    conns["weight_matrix"] = np.zeros((len(pop_post), len(pop_pre)))
    conns["weight_matrix"][conns["targets"], conns["senders"]] = conns["weight"]
    return conns

# ====================== SIMULACIÓN ======================
nest.Simulate(duration["sim"])

weights_post_train = {
    "in_rec": get_weights(nrns_in, nrns_rec),
    "rec_rec": get_weights(nrns_rec, nrns_rec),
    "rec_out": get_weights(nrns_rec, nrns_out),
}

# ====================== ANÁLISIS ======================
os.makedirs("output_estilos", exist_ok=True)

events_mm_out = mm_out.get("events")
readout_signal = events_mm_out["readout_signal"]
target_signal = events_mm_out["target_signal"]
senders = events_mm_out["senders"]

# IDs únicos de neuronas de salida (recalcular siempre aquí)
sender_ids = sorted(np.unique(senders))
readouts = [readout_signal[senders == sid] for sid in sender_ids]
targets = [target_signal[senders == sid] for sid in sender_ids]

seq_len = steps["sequence"]
n_sequences = n_iter * n_batch

# 📉 Calcular LOSS por neurona
loss_list = []
for sid in sender_ids:
    idc = senders == sid
    error = (readout_signal[idc] - target_signal[idc]) ** 2
    loss_list.append(0.5 * np.add.reduceat(error, np.arange(0, steps["task"], steps["sequence"])))
loss = np.sum(loss_list, axis=0)

# 📈 Graficar LOSS
colors = {
    "blue": "#2854c5ff",
    "red": "#e04b40ff",
    "white": "#ffffffff",
}
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.prop_cycle": plt.cycler(color=[colors["blue"], colors["red"]]),
})

fig, ax = plt.subplots()
ax.plot(range(1, len(loss_list[0]) + 1), loss_list[0], label=r"$E_0$", alpha=0.8, c=colors["blue"], ls="--")
ax.plot(range(1, len(loss_list[1]) + 1), loss_list[1], label=r"$E_1$", alpha=0.8, c=colors["blue"], ls="dotted")
ax.plot(range(1, len(loss) + 1), loss, label=r"$E$", c=colors["blue"])
ax.set_ylabel(r"$E = \frac{1}{2} \sum_{t,k} \left( y_k^t -y_k^{*,t}\right)^2$")
ax.set_xlabel("training iteration")
ax.set_xlim(1, len(loss))
ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
fig.tight_layout()
plt.savefig("output_estilos/loss_training.png", dpi=300)
plt.close()

# 📉 MSE GLOBAL
readout_0, readout_1 = readouts[0], readouts[1]
target_0, target_1 = targets[0], targets[1]
mse_global_x = mean_squared_error(target_0, readout_0)
mse_global_y = mean_squared_error(target_1, readout_1)
mse_global = (mse_global_x + mse_global_y) / 2
print(f"\n📈 MSE GLOBAL:\n - X: {mse_global_x:.6f}\n - Y: {mse_global_y:.6f}\n - Promedio: {mse_global:.6f}")

# ✅ Cargar combinaciones (letra, estilo) por secuencia
letras_estilos_por_secuencia = np.load("output_estilos/letras_estilos_por_secuencia.npy")

# 🗂️ Agrupar secuencias por (letra_idx, estilo_idx)
secuencias_por_letra_estilo = defaultdict(list)
for i, (letra_idx, estilo_idx) in enumerate(letras_estilos_por_secuencia):
    secuencias_por_letra_estilo[(letra_idx, estilo_idx)].append(i)

print("\n📊 MSE por combinación letra-estilo:")
for (letra_idx, estilo_idx), indices in secuencias_por_letra_estilo.items():
    last_idx = indices[-1]
    start = last_idx * seq_len
    end = start + seq_len
    mse_x = mean_squared_error(target_0[start:end], readout_0[start:end])
    mse_y = mean_squared_error(target_1[start:end], readout_1[start:end])
    mse_total = (mse_x + mse_y) / 2
    print(f" - Letra {letra_idx}, Estilo {estilo_idx}: MSE_X={mse_x:.6f}, MSE_Y={mse_y:.6f}, MSE={mse_total:.6f}")

# 📊 Visualizar última trayectoria aprendida de cada estilo
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for i, estilo_idx in enumerate([0, 1]):  # A1 y A2
    letra_idx = 0
    seq_id = secuencias_por_letra_estilo[(letra_idx, estilo_idx)][-1]  # última ocurrencia
    start = seq_id * seq_len
    end = start + seq_len
    y0_seq = readouts[0][start:end]
    y1_seq = readouts[1][start:end]
    axs[i].plot(y0_seq, -y1_seq, color="blue")
    axs[i].set_title(f"Estilo A{estilo_idx+1}")
    axs[i].axis("equal")
    axs[i].axis("off")
plt.tight_layout()
plt.show()

# ====================== GUARDAR MODELO Y VISUALIZACIONES ======================
output_folder = "interpolacion"
os.makedirs(output_folder, exist_ok=True)

modelo = {
    "pesos_in_rec": weights_post_train["in_rec"]["weight_matrix"],
    "pesos_rec_rec": weights_post_train["rec_rec"]["weight_matrix"],
    "pesos_rec_out": weights_post_train["rec_out"]["weight_matrix"],
    "n_in": n_in,
    "n_rec": n_rec,
    "n_out": n_out,
    "duration": duration,
    "steps": steps,
}
ruta_modelo = "interpolacion/interpolacion.pkl"
with open(ruta_modelo, "wb") as f:
    pickle.dump(modelo, f)
print(f"✅ Modelo guardado en: {ruta_modelo}")

# Leer outputs y secuencia (otra vez, ya consolidado)
events = mm_out.get("events")
senders2 = events["senders"]
sender_ids2 = sorted(np.unique(senders2))
readout_0b = np.array(events["readout_signal"][senders2 == sender_ids2[0]]).ravel()
readout_1b = np.array(events["readout_signal"][senders2 == sender_ids2[1]]).ravel()
target_0b  = np.array(events["target_signal"][senders2 == sender_ids2[0]]).ravel()
target_1b  = np.array(events["target_signal"][senders2 == sender_ids2[1]]).ravel()

letras_estilos_por_secuencia = np.load("output_estilos/letras_estilos_por_secuencia.npy")
seq_len = steps["sequence"]

# Agrupar últimas trayectorias por estilo
secuencias_por_estilo = defaultdict(list)
for i, (letra, estilo) in enumerate(letras_estilos_por_secuencia):
    secuencias_por_estilo[estilo].append(i)

# Visualizar y guardar
output_dir = "interpolacion"
os.makedirs(output_dir, exist_ok=True)

for estilo_idx in [0, 1]:
    last_seq_idx = secuencias_por_estilo[estilo_idx][-1]
    start = last_seq_idx * seq_len
    end = start + seq_len

    r0 = readout_0b[start:end]
    r1 = readout_1b[start:end]
    t0 = target_0b[start:end]
    t1 = target_1b[start:end]

    mse_x = mean_squared_error(t0, r0)
    mse_y = mean_squared_error(t1, r1)

    estilo_name = f"A{estilo_idx + 1}"
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(r0, -r1, label="readout", color="red")
    ax.plot(t0, -t1, label="target", color="blue")
    ax.set_title(f"Letra A estilo {estilo_name}\nMSE X: {mse_x:.4f} | Y: {mse_y:.4f}")
    ax.axis("equal")
    ax.axis("off")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/A_{estilo_name}.png", dpi=300)
    plt.close()

print("✅ Visualizaciones guardadas en la carpeta 'interpolacion'")
