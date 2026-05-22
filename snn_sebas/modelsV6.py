import torch
import torch.nn as nn


class LIF(nn.Module):
    def __init__(
        self,
        input_size,
        size,
        threshold=1.0,
        tau=0.9,
        is_recurrent=False,
        gamma=0.3,
        beta=1.0,
        refractory_steps=2,
    ):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.is_recurrent = is_recurrent
        self.gamma = gamma
        self.beta = beta
        self.refractory_steps = refractory_steps
        self.input_weights = nn.Linear(input_size, size, bias=False)
        if is_recurrent:
            self.recurrent_weights = nn.Linear(size, size, bias=False)

    def forward(self, x, v_old, z_old=None, refractory_count=None):
        v_new = v_old * self.tau + self.input_weights(x)
        if self.is_recurrent and z_old is not None:
            v_new = v_new + self.recurrent_weights(z_old)

        surrogate_grad = (self.gamma / self.threshold) * torch.clamp(
            1 - self.beta * abs((v_new - self.threshold) / self.threshold),
            min=0,
        )

        if refractory_count is None:
            refractory_count = torch.zeros_like(v_new)

        available = refractory_count <= 0
        v_new = torch.where(available, v_new, torch.zeros_like(v_new))

        spikes = ((v_new > self.threshold) & available).float()
        v_new = v_new - spikes * self.threshold

        refractory_count = torch.clamp(refractory_count - 1, min=0)
        refractory_count = torch.where(
            spikes > 0,
            torch.full_like(refractory_count, self.refractory_steps),
            refractory_count,
        )
        surrogate_grad = surrogate_grad * available.float()

        return spikes, v_new, surrogate_grad, refractory_count


class SNN(nn.Module):
    def __init__(
        self,
        i_size=2312,
        h_size=64,
        o_size=10,
        tau=0.9,
        tau_o=0.9,
        f_target=10.0,
        c_reg=0.0,
        use_reg=False,
        refractory_steps=2,
    ):
        super().__init__()
        self.hidden_size = h_size
        self.output_size = o_size
        self.tau = tau
        self.tau_o = tau_o
        self.f_target = f_target / 1000.0
        self.c_reg = c_reg
        self.use_reg = use_reg

        self.hidden_layer = LIF(
            i_size,
            h_size,
            is_recurrent=True,
            refractory_steps=refractory_steps,
        )
        self.readout_weights = nn.Linear(h_size, o_size, bias=False)
        self.register_buffer("B", torch.randn(h_size, o_size) * 0.1)

    def forward(self, x, labels=None):
        batch, T, _ = x.shape
        device = x.device
        training = labels is not None

        v_h = torch.zeros(batch, self.hidden_size, device=device)
        v_o = torch.zeros(batch, self.output_size, device=device)
        z_h = torch.zeros(batch, self.hidden_size, device=device)
        refractory_count = torch.zeros(batch, self.hidden_size, device=device)

        x_trace = torch.zeros(batch, x.shape[2], device=device)
        z_trace = torch.zeros(batch, self.hidden_size, device=device)
        f_avg = torch.zeros(batch, self.hidden_size, device=device)

        if training:
            grad_inp = torch.zeros_like(self.hidden_layer.input_weights.weight)
            grad_rec = torch.zeros_like(self.hidden_layer.recurrent_weights.weight)
            grad_out = torch.zeros_like(self.readout_weights.weight)
            y_target = torch.zeros(batch, self.output_size, device=device)
            y_target.scatter_(1, labels.unsqueeze(1), 1.0)

        with torch.no_grad():
            for t in range(T):
                x_t = x[:, t, :]
                h_spikes, v_h, surrogate_grad, refractory_count = self.hidden_layer(
                    x_t, v_h, z_h, refractory_count
                )
                v_o = self.tau_o * v_o + self.readout_weights(h_spikes)

                if training:
                    x_trace = self.tau * x_trace + x_t
                    z_trace = self.tau * z_trace + z_h

                    y_hat = torch.softmax(v_o, dim=1)
                    output_error = y_hat - y_target
                    learning_signal = output_error.matmul(self.B.t())

                    if self.use_reg and self.c_reg != 0.0:
                        f_avg = self.tau * f_avg + (1 - self.tau) * h_spikes
                        reg = self.c_reg * (self.f_target - f_avg)
                    else:
                        reg = 0.0

                    post_factor = (learning_signal + reg) * surrogate_grad
                    grad_inp += (
                        torch.einsum("bh,bi->hi", post_factor, x_trace) / batch
                    )
                    grad_rec += (
                        torch.einsum("bh,bi->hi", post_factor, z_trace) / batch
                    )
                    grad_out += (
                        torch.einsum("bo,bh->oh", output_error, z_trace) / batch
                    )

                z_h = h_spikes

        if training:
            self.hidden_layer.input_weights.weight.grad = grad_inp
            self.hidden_layer.recurrent_weights.weight.grad = grad_rec
            self.readout_weights.weight.grad = grad_out

        return v_o


