# DisAAD training patches

The DisAAD adversarial trainer is a git submodule (`third_party/DisAAD`); these are our
modified copies (uncertainty-aware modes, bf16 targets, resumable checkpointing, LoRA
r/alpha, batching). To reproduce training, clone the DisAAD submodule and overlay these
files, or run `scripts/mt_train.py` which drives them.

- `scripts/train_disaad.py` — SFT+adversarial(+uncertainty) trainer; `--uncertainty_mode {none,head,edl,both}`
- `scripts/rq5_uncertainty.py` — the uncertainty-aware loss (EDL target / MLP head vs. black-box oracle)
- `scripts/data_builder.py`, `scripts/model.py` — data + model plumbing
