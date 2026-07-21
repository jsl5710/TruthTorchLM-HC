"""Stage A -- generate and cache (protocol §7).

For each (item, target) draw the primary answer plus ``n_max`` samples once, and persist
them. This is the single most important control in the whole benchmark: every method and
every N in the sweep consumes this identical cache, so generation cost and generation
variance are removed as confounds between methods (protocol §6).

Maps onto DisAAD's ``generate_black.py`` (the "gene" pass), so results are comparable to
theirs.
"""

from tqdm import tqdm

from TruthTorchLM.templates import DEFAULT_SYSTEM_BENCHMARK_PROMPT, DEFAULT_USER_PROMPT
from TruthTorchLM.utils.dataset_utils import get_dataset

from .cache import GenerationCache

__all__ = ["generate_stage_a"]


def _build_messages(item, user_prompt, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    if item.get("context"):
        content = ("Context: {context}\n" + user_prompt).format(
            context=item["context"], question=item["question"]
        )
    else:
        content = user_prompt.format(question=item["question"])
    messages.append({"role": "user", "content": content})
    return messages


def generate_stage_a(
    config,
    seed: int,
    cache_root: str = "hc_benchmark/cache",
    user_prompt: str = DEFAULT_USER_PROMPT,
    system_prompt: str = DEFAULT_SYSTEM_BENCHMARK_PROMPT,
    overwrite: bool = False,
    _generate_fn=None,
    _sample_fn=None,
):
    """Produce and cache primary answer + ``n_max`` samples for one (config, seed) shard.

    ``_generate_fn`` / ``_sample_fn`` are injection points for testing without a live
    model: they default to the real API primitives. Only the ``litellm`` backend is wired
    here; the ``huggingface`` path raises with a pointer, because a local target also
    needs a loaded model + tokenizer handed in, which belongs to the caller's launch
    script rather than this orchestrator.
    """
    cache = GenerationCache(cache_root, config, seed)
    if cache.exists() and not overwrite:
        print(f"[Stage A] cache hit: {cache.path.name} ({len(cache)} items) -- skipping.")
        return cache

    if config.generator_backend == "litellm":
        generate_fn = _generate_fn or _default_api_generate
        sample_fn = _sample_fn or _default_api_sample
    else:
        raise NotImplementedError(
            "Stage A's huggingface backend needs a loaded model+tokenizer passed from the "
            "launch script. Use generate_stage_a_local(...) with those objects, or the "
            "litellm backend for API targets."
        )

    dataset = get_dataset(config.dataset, size_of_data=config.size_of_data, seed=seed)
    dec = config.decoding

    items = []
    for i, data in enumerate(tqdm(dataset, desc=f"[Stage A] {config.dataset}/{config.generator} seed{seed}")):
        messages = _build_messages(data, user_prompt, system_prompt)

        primary = generate_fn(
            model=config.generator, messages=messages,
            temperature=dec.temperature, top_p=dec.top_p, max_tokens=dec.max_tokens,
            seed=seed,
        )
        samples = sample_fn(
            model=config.generator, messages=messages, number_of_generations=config.n_max,
            temperature=dec.sample_temperature, top_p=dec.sample_top_p,
            max_tokens=dec.max_tokens, generation_seed=seed,
        )

        items.append(
            {
                "item_id": data.get("item_id", i),
                "question": data["question"],
                "context": data.get("context", ""),
                "ground_truths": data["ground_truths"],
                "primary_answer": primary,
                "samples": samples,
                "stratum": data.get("stratum"),
                "outcome_type": data.get("outcome_type", "factual_error"),
            }
        )

    cache.write(items)
    print(f"[Stage A] wrote {len(items)} items -> {cache.path.name}")
    return cache


def _default_api_generate(model, messages, **kwargs):
    from TruthTorchLM.generation import generate_api

    out = generate_api(model=model, messages=messages, **kwargs)
    return out["generated_text"]


def _default_api_sample(model, messages, number_of_generations, generation_seed, **kwargs):
    from TruthTorchLM.generation import sample_generations_api

    out = sample_generations_api(
        model=model, messages=messages, number_of_generations=number_of_generations,
        return_text=True, generation_seed=generation_seed, **kwargs
    )
    return out["generated_texts"] if out else []
