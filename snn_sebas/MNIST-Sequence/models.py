import torch
import torch.nn as nn
import torch.nn.functional as F


def make_hidden_parameter(size, value_range):
    low, high = value_range
    if low > high:
        raise ValueError(f"invalid parameter range: {value_range}")
    if low == high:
        return torch.full((size,), float(low))
    return torch.empty(size).uniform_(float(low), float(high))


def make_sparse_mask_like(weight, connectivity):
    if connectivity >= 1.0:
        return torch.ones_like(weight)
    if connectivity <= 0.0:
        return torch.zeros_like(weight)
    return (torch.rand_like(weight) < connectivity).float()


class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 5)),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 3, 1, 2).squeeze(-1)
        return self.classifier(x)


class CRNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.position_pool = nn.AdaptiveAvgPool1d(5)
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=2)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = x.transpose(1, 2)
        x = self.position_pool(x)
        x = x.transpose(1, 2)
        return self.classifier(x)


class TCNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        )
        self.position_pool = nn.AdaptiveAvgPool1d(5)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=2)
        x = self.temporal(x)
        x = self.position_pool(x)
        x = x.transpose(1, 2)
        return self.classifier(x)


VALID_NEURON_TYPES = {"lif", "alif", "mixed"}
VALID_FEEDBACK_TYPES = {"random", "symmetric"}
VALID_READOUT_MODES = {"filtered", "current"}


class RecurrentSpikingLayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        threshold_range=(1.0, 1.0),
        tau_mem_range=(0.9, 0.9),
        tau_adapt_range=(0.95, 0.95),
        adapt_scale_range=(0.0, 0.0),
        surrogate_gamma=0.3,
        surrogate_beta=1.0,
        refractory_steps=0,
        adaptive_mask=None,
        connectivity=1.0,
    ):
        super().__init__()
        self.surrogate_gamma = surrogate_gamma
        self.surrogate_beta = surrogate_beta
        self.refractory_steps = refractory_steps
        if adaptive_mask is None:
            adaptive_mask = torch.ones(hidden_size)
        self.register_buffer("adaptive_mask", adaptive_mask)
        self.register_buffer("threshold", make_hidden_parameter(hidden_size, threshold_range))
        self.register_buffer("tau_mem", make_hidden_parameter(hidden_size, tau_mem_range))
        self.register_buffer("tau_adapt", make_hidden_parameter(hidden_size, tau_adapt_range))
        self.register_buffer("adapt_scale", make_hidden_parameter(hidden_size, adapt_scale_range))
        self.input_weights = nn.Linear(input_size, hidden_size, bias=False)
        self.recurrent_weights = nn.Linear(hidden_size, hidden_size, bias=False)
        self.register_buffer(
            "recurrent_mask",
            make_sparse_mask_like(self.recurrent_weights.weight, connectivity),
        )

    def forward(self, x_t, voltage, adaptation, prev_spikes, refractory):
        masked_recurrent_weights = self.recurrent_weights.weight * self.recurrent_mask
        current = self.input_weights(x_t) + F.linear(prev_spikes, masked_recurrent_weights)
        threshold = self.threshold + self.adapt_scale * adaptation * self.adaptive_mask
        voltage = self.tau_mem.unsqueeze(0) * voltage + current
        safe_threshold = threshold.clamp_min(1e-6)
        surrogate_grad = (self.surrogate_gamma / safe_threshold) * torch.clamp(
            1.0 - self.surrogate_beta * torch.abs((voltage - threshold) / safe_threshold),
            min=0.0,
        )
        available = refractory <= 0
        voltage = torch.where(available, voltage, torch.zeros_like(voltage))
        spikes = ((voltage > threshold) & available).float()
        voltage = voltage - spikes * threshold
        adaptation = self.tau_adapt.unsqueeze(0) * adaptation + spikes * self.adaptive_mask
        refractory = torch.clamp(refractory - 1, min=0)
        refractory = torch.where(
            spikes > 0,
            torch.full_like(refractory, self.refractory_steps),
            refractory,
        )
        return spikes, voltage, adaptation, surrogate_grad, refractory


class SNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=96,
        output_size=10,
        neuron_type="mixed",
        threshold=1.0,
        tau=0.9,
        tau_o=0.9,
        tau_adapt=0.95,
        adapt_scale=0.2,
        lif_ratio=0.7,
        f_target=10.0,
        c_reg=0.0,
        use_reg=False,
        refractory_steps=0,
        feedback_scale=0.1,
        feedback_type="symmetric",
        readout_mode="filtered",
        connectivity=1.0,
        tau_range=None,
        tau_adapt_range=None,
        threshold_range=None,
        adapt_scale_range=None,
    ):
        super().__init__()
        neuron_type = neuron_type.lower()
        feedback_type = feedback_type.lower()
        readout_mode = readout_mode.lower()
        if neuron_type not in VALID_NEURON_TYPES:
            raise ValueError(f"neuron_type must be one of: {sorted(VALID_NEURON_TYPES)}")
        if feedback_type not in VALID_FEEDBACK_TYPES:
            raise ValueError(f"feedback_type must be one of: {sorted(VALID_FEEDBACK_TYPES)}")
        if readout_mode not in VALID_READOUT_MODES:
            raise ValueError(f"readout_mode must be one of: {sorted(VALID_READOUT_MODES)}")
        if not 0.0 <= lif_ratio <= 1.0:
            raise ValueError("lif_ratio must be between 0 and 1")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.neuron_type = neuron_type
        self.feedback_type = feedback_type
        self.readout_mode = readout_mode
        self.tau_mem = tau
        self.tau_o = tau_o
        self.tau_adapt = tau_adapt
        self.use_reg = use_reg
        self.c_reg = c_reg
        self.target_rate = f_target / 1000.0
        self.tau_range = (tau, tau) if tau_range is None else tuple(tau_range)
        self.tau_adapt_range = (tau_adapt, tau_adapt) if tau_adapt_range is None else tuple(tau_adapt_range)
        self.threshold_range = (threshold, threshold) if threshold_range is None else tuple(threshold_range)
        self.adapt_scale_range = (
            (0.0, 0.0) if neuron_type == "lif" else ((adapt_scale, adapt_scale) if adapt_scale_range is None else tuple(adapt_scale_range))
        )

        adaptive_mask = torch.ones(hidden_size)
        if neuron_type == "lif":
            adaptive_mask.zero_()
        elif neuron_type == "mixed":
            num_lif = int(round(hidden_size * lif_ratio))
            adaptive_mask[:num_lif] = 0.0

        self.hidden_layer = RecurrentSpikingLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            threshold_range=self.threshold_range,
            tau_mem_range=self.tau_range,
            tau_adapt_range=self.tau_adapt_range,
            adapt_scale_range=self.adapt_scale_range,
            refractory_steps=refractory_steps,
            adaptive_mask=adaptive_mask,
            connectivity=connectivity,
        )
        self.readout_weights = nn.Linear(hidden_size, output_size, bias=False)
        self.register_buffer("feedback_weights", torch.randn(hidden_size, output_size) * feedback_scale)

    def _feedback_matrix(self):
        if self.feedback_type == "symmetric":
            return self.readout_weights.weight.detach().t()
        return self.feedback_weights

    def forward(self, x, labels=None):
        batch, T, _ = x.shape
        device = x.device
        training = labels is not None

        voltage = torch.zeros(batch, self.hidden_size, device=device)
        adaptation = torch.zeros(batch, self.hidden_size, device=device)
        spikes_prev = torch.zeros(batch, self.hidden_size, device=device)
        refractory = torch.zeros(batch, self.hidden_size, device=device)
        readout_trace = torch.zeros(batch, self.hidden_size, device=device)
        logits_over_time = []

        elig_in = torch.zeros(batch, self.hidden_size, self.input_size, device=device)
        elig_rec = torch.zeros(batch, self.hidden_size, self.hidden_size, device=device)
        elig_in_adapt = torch.zeros_like(elig_in)
        elig_rec_adapt = torch.zeros_like(elig_rec)
        firing_rate = torch.zeros(batch, self.hidden_size, device=device)

        if training:
            grad_in = torch.zeros_like(self.hidden_layer.input_weights.weight)
            grad_rec = torch.zeros_like(self.hidden_layer.recurrent_weights.weight)
            grad_out = torch.zeros_like(self.readout_weights.weight)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1).expand(-1, T)

        with torch.no_grad():
            for t in range(T):
                x_t = x[:, t, :]
                spikes, voltage, adaptation, psi, refractory = self.hidden_layer(
                    x_t,
                    voltage,
                    adaptation,
                    spikes_prev,
                    refractory,
                )
                readout_trace = self.tau_o * readout_trace + spikes
                readout_source = readout_trace if self.readout_mode == "filtered" else spikes
                logits = self.readout_weights(readout_source)
                logits_over_time.append(logits)

                if training:
                    elig_in = self.tau_mem * elig_in + x_t.unsqueeze(1)
                    elig_rec = self.tau_mem * elig_rec + spikes_prev.unsqueeze(1)
                    if self.neuron_type in {"alif", "mixed"}:
                        adapt_scale = self.hidden_layer.adapt_scale.unsqueeze(0).unsqueeze(-1)
                        elig_in_adapt = self.tau_adapt * elig_in_adapt + psi.unsqueeze(-1) * elig_in
                        elig_rec_adapt = self.tau_adapt * elig_rec_adapt + psi.unsqueeze(-1) * elig_rec
                        effective_in = psi.unsqueeze(-1) * (
                            elig_in - adapt_scale * elig_in_adapt
                        )
                        effective_rec = psi.unsqueeze(-1) * (
                            elig_rec - adapt_scale * elig_rec_adapt
                        )
                    else:
                        effective_in = psi.unsqueeze(-1) * elig_in
                        effective_rec = psi.unsqueeze(-1) * elig_rec

                    y_target = torch.zeros(batch, self.output_size, device=device)
                    y_target.scatter_(1, labels[:, t].unsqueeze(1), 1.0)
                    probabilities = torch.softmax(logits, dim=1)
                    output_error = probabilities - y_target
                    learning_signal = output_error @ self._feedback_matrix().t()

                    if self.use_reg and self.c_reg != 0.0:
                        firing_rate = firing_rate + (spikes - firing_rate) / float(t + 1)
                        learning_signal = learning_signal + self.c_reg * (
                            self.target_rate - firing_rate
                        )

                    grad_in += torch.einsum("bh,bhi->hi", learning_signal, effective_in) / batch
                    grad_rec += torch.einsum("bh,bhi->hi", learning_signal, effective_rec) / batch
                    grad_out += torch.einsum("bo,bh->oh", output_error, readout_source) / batch

                spikes_prev = spikes

        if training:
            self.hidden_layer.input_weights.weight.grad = grad_in
            self.hidden_layer.recurrent_weights.weight.grad = grad_rec
            self.readout_weights.weight.grad = grad_out

        return torch.stack(logits_over_time, dim=1)
        

def build_model(name):
    registry = {
        "cnn": CNNBaseline,
        "crnn": CRNNBaseline,
        "tcn": TCNBaseline,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {', '.join(registry)}"
        )
    return registry[name]()


def sequence_loss(logits, targets):
    batch, seq_len, num_classes = logits.shape
    return F.cross_entropy(
        logits.reshape(batch * seq_len, num_classes),
        targets.reshape(batch * seq_len),
    )


def sequence_accuracy(logits, targets):
    preds = logits.argmax(dim=-1)
    digit_acc = (preds == targets).float().mean().item()
    seq_acc = (preds == targets).all(dim=1).float().mean().item()
    return digit_acc, seq_acc
