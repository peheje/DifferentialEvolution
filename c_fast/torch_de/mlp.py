"""Flat-parameter MLP evaluator without autograd.

Matches C genann weight layout:
  For each hidden neuron: [-bias, w0, w1, ..., w_{prev-1}]
  For each output neuron: [-bias, w0, w1, ..., w_{prev-1}]

The stored bias is negated; we undo the negation on extraction.

Default hidden activation: sigmoid (matching C genann default).
"""

import torch


def _build_pack_index(inputs, hidden_layers, hidden, outputs):
    """Pre-compute CPU index tensors for fast unpacking from flat vectors.

    Returns (row_idx_l, col_idx_l, w_idx_l, bias_idx_l) per layer on CPU.
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


# Cache: (arch, device) -> list of (r, c, wi, bi) tensors on that device
_pack_cache: dict = {}


def _get_pack_indices(inputs, hidden_layers, hidden, outputs, device):
    """Get pack index tensors for an architecture, cached per device.

    Returns (row_idx_l, col_idx_l, w_idx_l, bias_idx_l) as lists of tensors
    on the requested device.
    """
    arch_key = (inputs, hidden_layers, hidden, outputs)
    cache_key = (arch_key, device)
    if cache_key not in _pack_cache:
        if arch_key not in _pack_cache:
            _pack_cache[arch_key] = _build_pack_index(
                inputs, hidden_layers, hidden, outputs)
        rl_cpu, cl_cpu, wl_cpu, bl_cpu = _pack_cache[arch_key]
        if device == "cpu":
            _pack_cache[cache_key] = (rl_cpu, cl_cpu, wl_cpu, bl_cpu)
        else:
            _pack_cache[cache_key] = (
                [t.to(device) for t in rl_cpu],
                [t.to(device) for t in cl_cpu],
                [t.to(device) for t in wl_cpu],
                [t.to(device) for t in bl_cpu],
            )
    return _pack_cache[cache_key]


def unpack_flat(flat, inputs, hidden_layers, hidden, outputs):
    """Unpack flat parameter vector(s) into weight matrices and biases.

    For a single vector flat [D]: returns (Ws, Bs) lists.
    For batched flat [P, D]: returns (Ws, Bs) where Ws[k] is [P, out, in].
    """
    device = flat.device
    dtype = flat.dtype
    batched = flat.dim() == 2
    P = flat.shape[0] if batched else 1

    # Get cached indices on the right device
    rl, cl, wl, bl = _get_pack_indices(
        inputs, hidden_layers, hidden, outputs, str(device))

    Ws = []
    Bs = []

    for layer_idx in range(len(rl)):
        r = rl[layer_idx]
        c = cl[layer_idx]
        wi = wl[layer_idx]
        bi = bl[layer_idx]

        n_out = int(r.max().item()) + 1
        n_in = int(c.max().item()) + 1

        # Gather: pop[:, wi] directly indexes flat along last dim.
        # wi iterates in row-major order, so reshape to [n_out, n_in].
        if batched:
            W = flat[:, wi].view(P, n_out, n_in)
            B = -flat[:, bi]
        else:
            W = flat[wi].view(n_out, n_in)
            B = -flat[bi]

        Ws.append(W)
        Bs.append(B)

    return Ws, Bs


class MLP:
    """Flat-parameter MLP matching C genann behavior.

    Default hidden activation is sigmoid (matching C genann default).
    """

    def __init__(self, inputs, hidden_layers, hidden, outputs,
                 activation=None):
        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.hidden = hidden
        self.outputs = outputs
        self.activation = activation if activation is not None else torch.sigmoid

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
