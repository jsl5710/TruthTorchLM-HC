"""RQ5 — uncertainty-aware distillation helpers (our addition; imported by train_disaad.py).

Adds an explicit uncertainty-alignment loss to DisAAD's generator objective so the proxy's
uncertainty is SUPERVISED against a pure-black-box teacher signal u*(x) (built by
scripts/rq5_build_labels.py), instead of being left emergent (RQ1 showed the emergent EU
collapses OOD). Two representations, selected by --uncertainty-mode:

  * head : an MLP on the proxy's last hidden state regresses u*(x)  (deployed via the head)
  * edl  : the proxy's evidential EU (top-k Dirichlet, DisAAD's own read-out) is pushed toward
           u*(x)                                                       (deployed via EDL, as DisAAD)

u*(x) is min-max normalized to [0,1] per oracle in the label file; both read-outs are in [0,1],
so a plain MSE aligns them. Everything here is pure-torch (runs in the `disaad` venv).
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["load_umap", "UncertaintyHead", "edl_uncertainty", "uncertainty_loss", "setup"]


def load_umap(labels_path, oracle):
    """{prompt -> u*(x) in [0,1] or None} for the chosen oracle."""
    d = json.load(open(labels_path))
    u = d["oracles"][oracle]["u_norm"]
    return {p: (float(x) if isinstance(x, (int, float)) else None) for p, x in zip(d["prompts"], u)}


class UncertaintyHead(nn.Module):
    """MLP on a pooled hidden state -> scalar predicted uncertainty in [0,1]."""

    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, pooled):            # (B, H) -> (B,)
        return self.net(pooled).squeeze(-1)


def _masked_mean(h, attn):                # h:(B,T,H) attn:(B,T) -> (B,H)
    m = attn.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


def edl_uncertainty(logits, labels, topk=10):
    """Per-example evidential EPISTEMIC uncertainty in (0,1], mean over the response tokens.

    EU = k / (sum(relu(top-k logits)) + k) — DisAAD's own evidence-scarcity measure, the one RQ1
    found un-preserved OOD; supervising it directly is the point. Response tokens = labels != -100.
    """
    B = logits.shape[0]
    out = []
    for b in range(B):
        pos = (labels[b] != -100).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            out.append(logits.new_tensor(0.5)); continue
        top = logits[b, pos].topk(topk, dim=-1).values          # (n, k)
        a = torch.clamp(top, min=0.0)
        out.append((topk / (a.sum(-1) + topk)).mean())          # (n,) -> scalar
    return torch.stack(out)                                       # (B,)


def uncertainty_loss(mode, outputs, inputs, prompts, umap, head, topk=10):
    """MSE between the proxy's uncertainty read-out(s) and the black-box target u*(x). Skips
    prompts with no label. mode 'both' = head + EDL losses summed (hybrid)."""
    dev = outputs.logits.device
    u_star = torch.tensor([umap.get(p) if umap.get(p) is not None else float("nan") for p in prompts],
                          device=dev, dtype=torch.float32)
    mask = ~torch.isnan(u_star)
    if not mask.any():
        return outputs.logits.new_tensor(0.0)
    total = outputs.logits.new_tensor(0.0)
    if mode in ("head", "both"):
        pooled = _masked_mean(outputs.hidden_states[-1], inputs["attention_mask"])
        u_head = head(pooled.to(next(head.parameters()).dtype))
        total = total + F.mse_loss(u_head[mask].float(), u_star[mask])
    if mode in ("edl", "both"):
        eu = edl_uncertainty(outputs.logits, inputs["labels"], topk)
        total = total + F.mse_loss(eu[mask].float(), u_star[mask])
    return total


def setup(args, hidden_size, device, dtype):
    """Return (umap, head, needs_hidden_states). head is None for edl/none modes."""
    mode = getattr(args, "uncertainty_mode", "none")
    if mode == "none":
        return None, None, False
    umap = load_umap(args.uncertainty_labels, args.uncertainty_oracle)
    n_lab = sum(1 for v in umap.values() if v is not None)
    print(f"[rq5] uncertainty-aware: mode={mode} oracle={args.uncertainty_oracle} "
          f"lambda={args.uncertainty_lambda} labels={n_lab}")
    head = UncertaintyHead(hidden_size).to(device, dtype) if mode in ("head", "both") else None
    return umap, head, (mode in ("head", "both"))
