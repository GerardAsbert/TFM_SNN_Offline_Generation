"""
models.py — Adaptive LIF SNN for handwriting generation.

Architecture mirrors NEST's eprop handwriting network:
  input (n_in) --[W_in]--> recurrent adaptive LIF (n_rec) --[W_out]--> linear readout (n_out=3)

Training uses e-prop with feedback alignment:
  - Online gradient accumulation via eligibility traces (no BPTT graph stored)
  - Fixed random feedback matrix B replaces NEST's eprop_learning_signal_connection
  - Gradients are written directly to .grad; a standard Adam optimizer is used outside
"""

import math
import torch
import torch.nn as nn
import wandb


class AdaptiveLIFLayer(nn.Module):
    """
    Recurrent adaptive LIF layer corresponding to NEST's eprop_iaf_adapt_bsshslm_2020.

    Adaptive threshold:
        a[t]   = tau_a * a[t-1] + z[t-1]
        V_th[t] = threshold + adapt_beta * a[t]

    Membrane potential:
        v[t] = tau_m * v[t-1] + W_in * x[t] + W_rec * z[t-1]

    Spike + soft reset:
        z[t] = H(v[t] - V_th[t])   (H = Heaviside)
        v[t] = v[t] - z[t] * V_th[t]

    Surrogate gradient (piecewise linear, same as NEST):
        psi[t] = (gamma / threshold) * max(0, 1 - |v[t] - V_th[t]| / threshold)
    """

    def __init__(
        self,
        input_size: int,
        size: int,
        tau_m: float = 0.967,       # exp(-dt/tau_m_ms)
        tau_a: float = 0.9995,      # exp(-dt/tau_a_ms)
        threshold: float = 0.03,    # V_th (mV)
        adapt_beta: float = None,   # prefactor of adaptive threshold
        gamma: float = 0.3,         # surrogate gradient scaling
        is_recurrent: bool = True,
    ):
        super().__init__()
        self.size = size
        self.tau_m = tau_m
        self.tau_a = tau_a
        self.threshold = threshold
        self.gamma = gamma
        self.is_recurrent = is_recurrent

        if adapt_beta is None:
            adapt_beta = (
                1.7 * (1.0 - math.exp(-1.0 / 2000.0))
                / (1.0 - math.exp(-1.0 / 30.0))
            )
        self.adapt_beta = adapt_beta

        self.input_weights = nn.Linear(input_size, size, bias=False)
        if is_recurrent:
            self.recurrent_weights = nn.Linear(size, size, bias=False)

        self._init_weights(input_size)

    def _init_weights(self, input_size: int) -> None:
        with torch.no_grad():
            nn.init.normal_(self.input_weights.weight, std=1.0 / math.sqrt(input_size))
            if self.is_recurrent:
                nn.init.normal_(
                    self.recurrent_weights.weight, std=1.0 / math.sqrt(self.size)
                )
                self.recurrent_weights.weight.fill_diagonal_(0.0)  # no autapses

    def _surrogate_grad(self, v: torch.Tensor, v_th: torch.Tensor) -> torch.Tensor:
        return (self.gamma / self.threshold) * torch.clamp(
            1.0 - torch.abs(v - v_th) / self.threshold, min=0.0
        )

    def step(
        self,
        x_t: torch.Tensor,     # (batch, input_size)
        v: torch.Tensor,        # (batch, size)
        a: torch.Tensor,        # (batch, size)  adaptation variable
        z_prev: torch.Tensor,   # (batch, size)  spikes from previous step
    ):
        """Single timestep. Returns (z_new, v_new, a_new, surrogate_grad)."""
        a_new = self.tau_a * a + z_prev
        v_th = self.threshold + self.adapt_beta * a_new

        v_new = self.tau_m * v + self.input_weights(x_t)
        if self.is_recurrent:
            v_new = v_new + self.recurrent_weights(z_prev)

        sg = self._surrogate_grad(v_new, v_th)
        z_new = (v_new > v_th).float()
        v_new = v_new - z_new * v_th   # soft reset

        return z_new, v_new, a_new, sg


class HandwritingSNN(nn.Module):
    """
    SNN for handwriting trajectory regression using e-prop.

    Output channels (n_out=3): dx_cumulative, dy_cumulative, pen.

    Usage
    -----
    Training (e-prop, no autograd graph):
        optimizer.zero_grad()
        out = model(x, targets=t)   # gradients written to .grad internally
        optimizer.step()

    Inference:
        out = model(x)              # (batch, T, n_out)
    """

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int = 3,
        tau_m_ms: float = 30.0,
        tau_a_ms: float = 2000.0,
        tau_out_ms: float = 50.0,
        dt: float = 1.0,
        threshold: float = 0.03,
        gamma: float = 0.3,
        f_target: float = 20.0,   # Hz, for optional firing-rate regularisation
        c_reg: float = 0.0,       # regularisation coefficient (0 = off)
        learning_signal_mode: str = "symmetric",
    ):
        super().__init__()
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.f_target = f_target / 1000.0   # spikes per ms
        self.c_reg = c_reg
        self.learning_signal_mode = learning_signal_mode

        tau_m = math.exp(-dt / tau_m_ms)
        tau_a = math.exp(-dt / tau_a_ms)
        self.tau_out = math.exp(-dt / tau_out_ms)
        adapt_beta = (
            1.7 * (1.0 - math.exp(-dt / tau_a_ms))
            / (1.0 - math.exp(-dt / tau_m_ms))
        )

        self.hidden_layer = AdaptiveLIFLayer(
            n_in, n_rec,
            tau_m=tau_m, tau_a=tau_a,
            threshold=threshold, adapt_beta=adapt_beta, gamma=gamma,
        )
        self.readout = nn.Linear(n_rec, n_out, bias=False)
        nn.init.normal_(self.readout.weight, std=1.0 / math.sqrt(n_rec))


        if learning_signal_mode == "random":
            self.register_buffer(
                "B",
                torch.randn(n_rec, n_out) / math.sqrt(n_rec)
            )

        elif learning_signal_mode == "adaptive":
            self.B = nn.Parameter(
                torch.randn(n_rec, n_out) / math.sqrt(n_rec)
            )


    def forward(self, x: torch.Tensor, targets: torch.Tensor = None, log_step: int = None) -> torch.Tensor:
        """
        Args
        ----
        x       : (batch, T, n_in)   binary spike trains (float32)
        targets : (batch, T, n_out)  regression targets; supply only during training

        Returns
        -------
        outputs : (batch, T, n_out)  readout membrane potentials
        """
        batch, T, _ = x.shape
        device = x.device
        training = targets is not None

        v_h = torch.zeros(batch, self.n_rec, device=device)
        a_h = torch.zeros(batch, self.n_rec, device=device)
        z_h = torch.zeros(batch, self.n_rec, device=device)
        v_out = torch.zeros(batch, self.n_out, device=device)

        tau_trace = self.hidden_layer.tau_m
        #x_trace = torch.zeros(batch, self.n_in, device=device)
        #z_trace = torch.zeros(batch, self.n_rec, device=device)
        f_avg = torch.zeros(batch, self.n_rec, device=device)

        # Eligibility traces (epsilons actually) for input neurons
        eps_v_in = torch.zeros(batch, self.n_rec, self.n_in, device=device)
        eps_a_in = torch.zeros(batch, self.n_rec, self.n_in, device=device)
        # Eligibility traces (epsilons actually) for recurrent neurons
        eps_v_rec = torch.zeros(batch, self.n_rec, self.n_rec, device=device)
        eps_a_rec = torch.zeros(batch, self.n_rec, self.n_rec, device=device)
        # Eligibility trace for output neurons
        z_out_trace = torch.zeros(batch, self.n_rec, device=device)


        if training:
            grad_inp = torch.zeros_like(self.hidden_layer.input_weights.weight)
            grad_rec = torch.zeros_like(self.hidden_layer.recurrent_weights.weight)
            grad_out = torch.zeros_like(self.readout.weight)

            if self.learning_signal_mode == "adaptive":
                grad_B = torch.zeros_like(self.B)

        outputs = []

        with torch.no_grad():
            for t in range(T):
                x_t = x[:, t, :]


                z_h_new, v_h, a_h, sg = self.hidden_layer.step(x_t, v_h, a_h, z_h)
                v_out = self.tau_out * v_out + self.readout(z_h_new)
                outputs.append(v_out.clone())

                if training:
                    
                    # Prepare necessary shapes
                    x_pre = x_t.unsqueeze(1)
                    z_pre = z_h.unsqueeze(1)
                    psi = sg.unsqueeze(-1)

                    # Eligibility traces (ET) (low-pass filtered pre-synaptic activity)
                    # epsilon regarding voltage of input neurons
                    eps_v_in_new = (
                        self.hidden_layer.tau_m * eps_v_in
                        - self.hidden_layer.adapt_beta * psi * eps_a_in
                        + x_pre
                    )
                    # epsilon regarding adaptation of input neurons
                    eps_a_in_new = (
                        psi * eps_v_in_new
                        + self.hidden_layer.tau_a * eps_a_in
                    )
                    # epsilon regarding voltage of recurrent neurons
                    eps_v_rec_new = (
                        self.hidden_layer.tau_m * eps_v_rec
                        - self.hidden_layer.adapt_beta * psi * eps_a_rec
                        + z_pre
                    )
                    # epsilon regarding adaptation of recurrent neurons
                    eps_a_rec_new = (
                        psi * eps_v_rec_new
                        + self.hidden_layer.tau_a * eps_a_rec
                    )

                    # ET for input neurons
                    e_in = psi * (
                        eps_v_in_new
                        - self.hidden_layer.adapt_beta * eps_a_in_new
                    )
                    # ET for recurrent neurons
                    e_rec = psi * (
                        eps_v_rec_new
                        - self.hidden_layer.adapt_beta * eps_a_rec_new
                    )
                    # ET for output neurons
                    z_out_trace = self.tau_out * z_out_trace + z_h_new

                    output_error = v_out - targets[:, t, :]           # (batch, n_out)

                    if log_step is not None and t % 10 == 0:
                        v_list   = v_out.detach().tolist()
                        tgt_list = targets[:, t, :].detach().tolist()
                        with open("debug_vectors.txt", "a") as _f:
                            _f.write(f"step={log_step} t={t}\n")
                            _f.write(f"  v_out:   {v_list}\n")
                            _f.write(f"  targets: {tgt_list}\n")

                    if self.learning_signal_mode == "adaptive":
                        
                        feedback_target = self.readout.weight.t()
                        grad_B += (
                            self.B - feedback_target
                        ) / batch

                    learning_signal = self.compute_learning_signal(output_error)

                    if self.c_reg > 0.0:
                        f_avg = tau_trace * f_avg + (1.0 - tau_trace) * z_h_new
                        learning_signal = learning_signal + self.c_reg * (self.f_target - f_avg)

                    #post_factor = learning_signal * sg                 # (batch, n_rec)
                    grad_inp += torch.einsum("bj,bji->ji", learning_signal, e_in) / batch
                    grad_rec += torch.einsum("bj,bji->ji", learning_signal, e_rec) / batch
                    grad_out += torch.einsum("bo,bh->oh", output_error, z_out_trace) / batch

                    eps_v_in = eps_v_in_new
                    eps_a_in = eps_a_in_new

                    eps_v_rec = eps_v_rec_new
                    eps_a_rec = eps_a_rec_new

                z_h = z_h_new

        if training:
            self.hidden_layer.input_weights.weight.grad = grad_inp
            self.hidden_layer.recurrent_weights.weight.grad = grad_rec
            self.readout.weight.grad = grad_out

            if self.learning_signal_mode == "adaptive":
                self.B.grad = grad_B

        return torch.stack(outputs, dim=1)  # (batch, T, n_out)
    

    def compute_learning_signal(
        self,
        output_error: torch.Tensor,
    ) -> torch.Tensor:

        if self.learning_signal_mode == "symmetric":
            return output_error.matmul(self.readout.weight)

        elif self.learning_signal_mode == "random":
            return output_error.matmul(self.B.t())

        elif self.learning_signal_mode == "adaptive":
            return output_error.matmul(self.B.t())

        else:
            raise ValueError(
                f"Unknown learning_signal_mode: {self.learning_signal_mode}"
            )
