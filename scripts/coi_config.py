"""CoI-Verbalized study config: models (G) and datasets (D).

Includes BOTH the paper-spec set (from the method MD) AND everything we already have
downloaded + cached from the benchmark, so runs can reuse cached generations (no Stage-A
regeneration) wherever possible. `READY_NOW` = (generator, dataset) cells whose Stage-A
cache already exists -> CoI-Verbalized scores them immediately (judge = the generator).
"""

# ---- Models (G = generator/target = judge) --------------------------------------
MODELS = {
    # key -> (HF repo, status).  status: 'cached' = we have Stage-A generations; 'weights' =
    # downloaded but no generations yet; 'spec' = named in the method MD, not local.
    "llama-3.1-8b":  ("meta-llama/Llama-3.1-8B-Instruct", "cached"),   # MD model #1, cached
    "qwen3-8b":      ("Qwen/Qwen3-8B",                     "cached"),   # substitutes MD's Qwen2-7B
    "qwen3-1.7b":    ("Qwen/Qwen3-1.7B",                   "cached"),
    "llama-3.2-3b":  ("meta-llama/Llama-3.2-3B-Instruct",  "cached"),
    "llama-3.2-1b":  ("meta-llama/Llama-3.2-1B-Instruct",  "cached"),
    "mistral-7b":    ("mistralai/Mistral-7B-Instruct-v0.3","cached"),
    "qwen3-32b":     ("Qwen/Qwen3-32B",                    "cached"),   # larger, mt cache (seed 0)
    "llama3.3-70b":  ("meta-llama/Llama-3.3-70B-Instruct", "cached"),
    # -- MD spec, not local (add only if the user wants exact-MD fidelity) --
    "qwen2-7b":      ("Qwen/Qwen2-7B-Instruct",            "spec"),     # MD model #2 (needs download)
}

# ---- Datasets (D) ----------------------------------------------------------------
# key -> (length, hop, format, area, cache_root, status)
DATASETS = {
    "trivia_qa":         ("short",     "single", "free", "general", "cache_full",        "cached"),  # MD core-pair #1
    "medlfqa":           ("long",      "-",      "free", "health",  "cache_full_health", "cached"),  # long-form -> substitutes MD LongFact
    "medqa":             ("short",     "single", "mcq",  "health",  "cache_full_health", "cached"),  # MD MedMCQA/MedQA
    "gsm8k":             ("short-ans", "multi",  "free", "math",    "cache_mt",          "cached"),  # MD GSM8K (mt targets)
    "truthful_qa":       ("short",     "single", "free", "adversar","cache_full",        "cached"),
    "natural_qa":        ("short",     "single", "free", "general", "cache_full",        "cached"),
    "pop_qa":            ("short",     "single", "free", "general", "cache_full",        "cached"),
    "mmlu_med":          ("short",     "single", "mcq",  "health",  "cache_full_health", "cached"),
    "bioasq":            ("short",     "single", "free", "health",  "cache_full_health", "cached"),
    # -- MD spec, need loaders/generation --
    "longfact":          ("long",      "-",      "free", "general", None,                "spec"),    # MD core-pair #2 (SAFE)
    "hotpot_qa":         ("short-med", "multi",  "free", "general", None,                "spec"),    # MD hop axis
}

# ---- Ready-now plan (reuse cached generations) -----------------------------------
# Core pair for the short-vs-long ablation, both cached open models:
CORE_MODELS   = ["llama-3.1-8b", "qwen3-8b"]
CORE_DATASETS = ["trivia_qa", "medlfqa"]          # short + long-form (both cached)
EXTEND_DATASETS = ["medqa", "gsm8k", "truthful_qa"]  # format=mcq, math, adversarial

CHAIN_COUNTS = [1, 2, 3, 4, 5]                     # Section-6 ablation rungs
