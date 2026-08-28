from .correctness_evaluator import CorrectnessEvaluator
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from litellm import completion
import random
import torch
from TruthTorchLM.templates import DEFAULT_JUDGE_PROMPT, DEFAULT_JUDGE_SYSTEM_PROMPT


class ModelJudge(CorrectnessEvaluator):
    """LLM-as-judge correctness evaluator.

    Two backends behind one interface:

    * ``model`` is a **string** -> routed through LiteLLM (an API judge, e.g. gpt-4o-mini).
    * ``model`` is a **loaded HF model** (+ ``tokenizer``) -> judged locally, no API. This
      is the path the open, key-free stages use, and it is hardened for a *small* local
      judge: the chat is built with an explicit generation prompt, decoding is greedy and
      length-bounded (so the same input reproducibly yields the same verdict), and the
      forward pass runs under ``inference_mode`` (no autograd graph).

    Verdicts map to TruthTorchLM's ints: ``1`` correct, ``0`` incorrect, ``-1``
    not-attempted / not-scorable. When the judge emits something that matches none of the
    expected words, the label is ``-1`` **and** ``last_verdict == "unparsed"`` -- the raw
    text is retained on ``last_raw_output`` so a caller can tell a real INCORRECT (label 0)
    apart from judge garbage (label -1, unparsed) and persist the offending output.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, str],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        prompt: str = DEFAULT_JUDGE_PROMPT,
        system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT,
        num_retries: int = 1,
        max_new_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.num_retries = num_retries
        self.max_new_tokens = max_new_tokens
        # Diagnostics from the most recent __call__, for a caller to inspect / persist:
        self.last_raw_output = None   # the judge's decoded verdict text
        self.last_verdict = None      # "correct" | "incorrect" | "not_attempted" | "unparsed"

    def _build_chat(self, question_text, generated_text, ground_truths, context):
        fields = dict(
            question=question_text,
            ground_truths=", ".join(ground_truths),
            answer=generated_text,
        )
        # The prompt template only interpolates {context} when it references it.
        if self.prompt.find("context") != -1:
            fields["context"] = context
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt.format(**fields)},
        ]

    @staticmethod
    def _parse_verdict(text: str):
        """Map raw judge text -> (verdict_name, int_label).

        "incorrect" is checked before "correct" because it *contains* "correct". Both
        not-attempted spellings the prompt uses ("NOT_ATTEMPTED" / "NOT ATTEMPTED") are
        accepted. Anything else is reported as "unparsed" -- so the caller can keep the raw
        output rather than silently folding garbage into not-attempted.
        """
        t = (text or "").lower()
        if "incorrect" in t:
            return "incorrect", 0
        if "correct" in t:
            return "correct", 1
        if "not_attempted" in t or "not attempted" in t or "not-attempted" in t:
            return "not_attempted", -1
        return "unparsed", -1

    def __call__(
        self,
        question_text: str,
        generated_text: str,
        ground_truths: list[str],
        context: str = "",
        seed: int = None,
    ) -> int:
        if seed is None:
            seed = random.randint(0, 1000000)
        chat = self._build_chat(question_text, generated_text, ground_truths, context)

        if isinstance(self.model, str):
            response = completion(
                model=self.model, messages=chat, seed=seed, num_retries=self.num_retries
            )
            judge_output = response.choices[0].message["content"]
        else:
            # Deterministic, length-bounded local judging. Greedy decoding makes the label
            # reproducible; add_generation_prompt cues the assistant turn so the model
            # emits a verdict instead of continuing the prompt; inference_mode drops the
            # autograd graph. Seeds are set for parity with the API path even though greedy
            # decoding is already deterministic.
            torch.manual_seed(seed)
            random.seed(seed)
            # return_dict=True yields a BatchEncoding carrying input_ids AND attention_mask;
            # transformers>=5 returns this dict-like object (not a bare tensor) for chat
            # templates, so build the mask from it rather than assuming a plain tensor.
            encoded = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(self.model.device)
            input_len = encoded["input_ids"].shape[1]
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            with torch.inference_mode():
                model_output = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=pad_id,
                )
            tokens = model_output[0][input_len:]
            judge_output = self.tokenizer.decode(tokens, skip_special_tokens=True)

        verdict, label = self._parse_verdict(judge_output)
        self.last_raw_output = judge_output
        self.last_verdict = verdict
        if verdict == "unparsed":
            print(
                "[ModelJudge] unparseable verdict -> labeling not_attempted (-1). "
                f"Raw output retained: {judge_output!r}"
            )
        return label

    def __str__(self):
        backend = self.model if isinstance(self.model, str) else type(self.model).__name__
        return f"ModelJudge(model={backend})"
