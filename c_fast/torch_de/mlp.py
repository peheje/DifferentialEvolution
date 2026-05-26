"""Flat-parameter MLP evaluator without autograd.

Matches C genann weight layout:
  For each hidden neuron: [-bias, w0, w1, ..., w_{prev-1}]
  For each output neuron: [-bias, w0, w1, ..., w_{prev-1}]

The stored bias is negated; we undo the negation on extraction.
"""

import torch


def _build_pack_index(inputs, hidden_layers, hidden, outputs):
    """Pre-compute index tensors for fast unpacking from flat vectors.

    Returns (row_idx_l, col_idx_l, w_idx_l, bias_idx_l) per layer.
    These index tensors live on CPU and are moved to device as needed.
    """
    row_idx_l = []
    col_idx_l = []
    w_idx_l = []
    bias_idx_l = []

    idx = 0
    for h in range(hidden_layers):
        prev = inputs if h == 0 else hidden
        rows = []
        cols = []
        w_idx = []
        b_idx = []
        for j in range(hidden):
            b_idx.append(idx)
            idx += 1
            for k in range(prev):
                rows.append(j)
                cols.append(k)
                w_idx.append(idx)
                idx += 1
        row_idx_l.append(torch.tensor(rows, dtype=torch.long))
        col_idx_l.append(torch.tensor(cols, dtype=torch.long))
        w_idx_l.append(torch.tensor(w_idx, dtype=torch.long))
        bias_idx_l.append(torch.tensor(b_idx, dtype=torch.long))

    # output layer
    out_prev = hidden if hidden_layers > 0 else inputs
    rows = []
    cols = []
    w_idx = []
    b_idx = []
    for o in range(outputs):
        b_idx.append(idx)
        idx += 1
        for k in range(out_prev):
            rows.append(o)
            cols.append(k)
            w_idx.append(idx)
            idx += 1
    row_idx_l.append(torch.tensor(rows, dtype=torch.long))
    col_idx_l.append(torch.tensor(cols, dtype=torch.long))
    w_idx_l.append(torch.tensor(w_idx, dtype=torch.long))
    bias_idx_l.append(torch.tensor(b_idx, dtype=torch.long))

    return row_idx_l, col_idx_l, w_idx_l, bias_idx_l


_pack_cache: dict = {}


def unpack_flat(flat, inputs, hidden_layers, hidden, outputs):
    """Unpack flat parameter vector(s) into weight matrices and biases.

    For a single vector flat [D]: returns (Ws, Bs) lists.
    For batched flat [P, D]: returns (Ws, Bs) where Ws[k] is [P, out, in].
    """
    cache_key = (inputs, hidden_layers, hidden, outputs)
    if cache_key not in _pack_cache:
        _pack_cache[cache_key] = _build_pack_index(
            inputs, hidden_layers, hidden, outputs)

    row_idx_l, col_idx_l, w_idx_l, bias_idx_l = _pack_cache[cache_key]
    device = flat.device
    dtype = flat.dtype
    batched = flat.dim() == 2
    P = flat.shape[0] if batched else 1

    Ws = []
    Bs = []

    for layer_idx in range(len(row_idx_l)):
        r = row_idx_l[layer_idx].to(device)
        c = col_idx_l[layer_idx].to(device)
        wi = w_idx_l[layer_idx].to(device)
        bi = bias_idx_l[layer_idx].to(device)

        n_out = int(r.max().item()) + 1
        n_in = int(c.max().item()) + 1

        if batched:
            W = torch.zeros(P, n_out, n_in, dtype=dtype, device=device)
            W[:, r, c] = flat[:, wi]
            B = -flat[:, bi]
        else:
            W = torch.zeros(n_out, n_in, dtype=dtype, device=device)
            W[r, c] = flat[wi]
            B = -flat[bi]

        Ws.append(W)
        Bs.append(B)

    return Ws, Bs


class MLP:
    """Flat-parameter MLP matching C genann behavior."""

    def __init__(self, inputs, hidden_layers, hidden, outputs,
                 activation=None):
        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.hidden = hidden
        self.outputs = outputs
        self.activation = activation if activation is not None else torch.tanh

        n = 0
        for h in range(hidden_layers):
            prev = inputs if h == 0 else hidden
            n += hidden * (1 + prev)
        out_prev = hidden if hidden_layers > 0 else inputs
        n += outputs * (1 + out_prev)
        self.total_weights = n

    def forward(self, flat, inputs):
        """Forward pass for a single flat vector [D].

        Returns [S, O].
        """
        Ws, Bs = unpack_flat(flat, self.inputs, self.hidden_layers,
                             self.hidden, self.outputs)
        x = inputs
        for k in range(len(Ws) - 1):
            x = self.activation(x @ Ws[k].T + Bs[k])
        return x @ Ws[-1].T + Bs[-1]

    def population_forward(self, pop, inputs):
        """Evaluate all population members [P, D] against inputs [S, I].

        Returns [P, S, O].
        """
        Ws, Bs = unpack_flat(pop, self.inputs, self.hidden_layers,
                             self.hidden, self.outputs)
        # Ws[k]: [P, n_out, n_in], Bs[k]: [P, n_out]

        P = pop.shape[0]
        # First layer: expand 2D input to batched 3D
        x = inputs.T.unsqueeze(0).expand(P, -1, -1)  # [P, I, S]

        for k in range(len(Ws)):
            # Ws[k]: [P, n_out, n_in], x: [P, n_in, S]
            x = torch.bmm(Ws[k], x)                 # [P, n_out, S]
            x = x + Bs[k].unsqueeze(-1)              # [P, n_out, S]
            x = x.permute(0, 2, 1)                   # [P, S, n_out]
            if k < len(Ws) - 1:
                x = self.activation(x)
                x = x.permute(0, 2, 1)               # [P, n_out, S] for next bmm

        return x  # [P, S, O]

    def to(self, device):
        return self
