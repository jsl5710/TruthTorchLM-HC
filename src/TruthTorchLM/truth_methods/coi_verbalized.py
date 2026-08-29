"""CoI-Verbalized: Chain-of-Interaction Verbalized Confidence.

Black-box UQ that judges a FIXED candidate answer y for query x without rewriting it.
The LLM generates structured, falsifiable evidence (decompose -> evidence/critique ->
consistency); the confidence SCORE is computed deterministically in code (phi), not
self-rated -- removing the same-pass overconfidence that plagues plain verbalized
confidence. Adapted from Chain-of-Interactions (Lucas et al., Findings of EMNLP 2025).

One TruthMethod, `chain_count in {1..7}` selects the ablation rung (Section 6):
  n=1  plain verbalized confidence (LLM self-rates y globally)            [the floor]
  n=2  decompose -> per-claim self-rated confidence, phi min/mean pooling
  n=3  decompose -> evidence+critique -> consistency; phi computes score  [DEFAULT]
  n=4  as n=3 but adversarial critique is its own separate chain
  n=5  as n=4 plus a final LLM self-rating over the full structure (reports both)
  n=6  decompose -> per-claim DECISION-framed YES/NO verify (fresh greedy pass), phi pools
       the YES-fractions. Replaces the hedgeable self-rated risk with a discriminative call.
  n=7  as n=6 but each claim is verified K times at T>0; confidence = vote fraction. A
       SELF-CONSISTENCY signal: hallucinated facts are unstable across samples, so vote
       entropy discriminates where a single self-rating (n=3 risk) cannot (medlfqa fix).

Returns {"truth_value": aggregate_confidence, ...raw chains, phi output...}. truth_value
is P(correct)-oriented (higher = more confident), matching VerbalizedConfidence.
"""
import copy
import json
import re
from typing import Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from .truth_method import TruthMethod
from ..generation import generate
from ..utils import fix_tokenizer_chat

# v0 fixed priors -- no training data needed to run (Section 4).
BASE = {"contextual": 1.00, "parametric": 0.85, "inferred": 0.65}
CONTRADICTION_PENALTY = 0.5
MIN_POOL_THRESHOLD = 0.5

SYSTEM = ("You are a meticulous, skeptical fact-checker. You judge how reliable a candidate "
          "answer is; you never rewrite or improve it. You output only a single JSON object.")

# ---- prompt bodies per rung (chains 1-3/4; aggregation is code, not generated) ----
_SCHEMA_N3 = '''{
  "chain_1_decompose": {"claims": [{"id": "c1", "text": "..."}]},
  "chain_2_evidence_critique": {"claims": [{"id": "c1", "grounding": "contextual|parametric|inferred", "evidence": "...", "counter_argument": "...", "risk": 0.0}]},
  "chain_3_consistency": {"contradictions": [["c1","c2"]], "answers_query": true, "drift_note": ""}
}'''

_PROMPT_N3 = '''You are given a query (x) and a candidate answer (y) produced by another model.
Assess how reliable y is. Do not rewrite or improve y -- only judge it.
Follow the 3-chain process and return exactly one JSON object matching the schema. Every key must be present and filled.

query: {x}
answer: {y}

Chain 1 -- Decompose: break y into atomic, independently checkable claims. A short factual answer -> exactly one claim (the full answer text). A long-form answer -> one claim per checkable assertion. Give each a unique id "c1","c2",... in order.
Chain 2 -- Evidence and critique (per claim): grounding in [contextual (supported by given input), parametric (model's trained knowledge), inferred (an inferential leap not stated)]; evidence (one sentence of support); counter_argument (as a skeptical fact-checker, the single strongest reason the claim could be wrong/outdated/ambiguous); risk (float 0.0-1.0, higher = more likely wrong, scaled to the counter_argument's strength).
Chain 3 -- Consistency scan: contradictions (list of claim-id pairs that conflict, empty if none); answers_query (boolean: does y address what x asked); drift_note (one sentence if answers_query is false/partial, else "").

Return only the JSON object below, fully filled in. No text outside the JSON. Do NOT include a confidence score or aggregation -- that is computed separately.
''' + _SCHEMA_N3

_PROMPT_N1 = '''You are given a query (x) and a candidate answer (y) from another model.
Judge how likely y is correct. Return only this JSON: {"confidence": <float 0-100>}.
query: {x}
answer: {y}'''

_SCHEMA_N2 = '''{
  "chain_1_decompose": {"claims": [{"id": "c1", "text": "..."}]},
  "chain_2_confidence": {"claims": [{"id": "c1", "confidence": 0.0}]}
}'''
_PROMPT_N2 = '''Given query (x) and a candidate answer (y), decompose y into atomic checkable claims (ids "c1","c2",...), then rate each claim's probability of being correct (confidence 0.0-1.0). Do not rewrite y. Return only this JSON:
query: {x}
answer: {y}
''' + _SCHEMA_N2

_SCHEMA_N4 = '''{
  "chain_1_decompose": {"claims": [{"id": "c1", "text": "..."}]},
  "chain_2a_evidence": {"claims": [{"id": "c1", "grounding": "contextual|parametric|inferred", "evidence": "..."}]},
  "chain_2b_critique": {"claims": [{"id": "c1", "counter_argument": "...", "risk": 0.0}]},
  "chain_3_consistency": {"contradictions": [["c1","c2"]], "answers_query": true, "drift_note": ""}
}'''
_PROMPT_N4 = '''You are given a query (x) and candidate answer (y). Do not rewrite y -- judge it.
Chain 1 Decompose into atomic claims (ids c1,c2,...).
Chain 2a Evidence: for each claim give grounding (contextual|parametric|inferred) and one evidence sentence.
Chain 2b Critique (a SEPARATE adversarial pass, having already gathered evidence): for each claim, the strongest counter_argument and a risk float 0.0-1.0.
Chain 3 Consistency: contradictions (claim-id pairs), answers_query (bool), drift_note.
Return only this JSON, fully filled, no text outside it:
query: {x}
answer: {y}
''' + _SCHEMA_N4

_PROMPT_N5_SUFFIX = '''\n\nAfter the above, having walked through your own structured critique and seen the
computed aggregate {phi_score:.2f}, give your OWN final calibrated confidence for y as a whole.
Return only this JSON: {{"self_confidence": <float 0-100>, "rationale": "..."}}'''

# ---- decision-framed verify rungs (n=6 greedy, n=7 self-consistency) ----
_PROMPT_DECOMPOSE = '''You are given a query (x) and a candidate answer (y) from another model.
Break y into atomic, independently checkable factual claims. If y is a short factual answer, output exactly ONE claim whose text is the full answer. If y is long-form, output one claim per checkable assertion, in order.
Return only this JSON, nothing else: {"claims": [{"id": "c1", "text": "..."}]}
query: {x}
answer: {y}'''

# Fresh, discriminative frame -- NOT "rate your risk 0-1" (which hedges to 0.5). One decision word.
_PROMPT_VERIFY = '''Question: {x}
Statement: {claim}
Is the statement factually correct and a correct answer to the question? Answer with exactly one word: YES or NO.'''


def _parse_yesno(text: str):
    """First standalone YES/NO in the completion -> 1.0 / 0.0; neither -> 0.5 (abstain)."""
    if not text:
        return 0.5
    m = re.search(r'\b(yes|no)\b', text, flags=re.IGNORECASE)
    if not m:
        return 0.5
    return 1.0 if m.group(1).lower() == "yes" else 0.0


def _extract_json(text: str):
    """Robustly pull the first balanced JSON object out of a model completion."""
    if not text:
        return None
    # strip code fences
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    frag = re.sub(r",\s*([}\]])", r"\1", text[start:i + 1])
                    try:
                        return json.loads(frag)
                    except json.JSONDecodeError:
                        break
    # never balanced -> generation was TRUNCATED (max_new_tokens hit mid-object). Salvage the
    # completed claims: cut back to the last complete "}", then close any still-open [ or {.
    frag = text[start:]
    cut = frag.rfind("}")
    if cut < 0:
        return None
    frag = frag[:cut + 1]
    opens = []
    for c in frag:
        if c in "{[":
            opens.append(c)
        elif c == "}" and opens and opens[-1] == "{":
            opens.pop()
        elif c == "]" and opens and opens[-1] == "[":
            opens.pop()
    frag = frag + "".join("]" if o == "[" else "}" for o in reversed(opens))
    frag = re.sub(r",\s*([}\]])", r"\1", frag)
    try:
        return json.loads(frag)
    except json.JSONDecodeError:
        return None


def _score_claim(grounding, risk, claim_id, contradictions):
    base = BASE.get((grounding or "parametric").lower(), BASE["parametric"])
    penalty = CONTRADICTION_PENALTY if any(claim_id in pair for pair in contradictions) else 1.0
    try:
        risk = float(risk)
    except (TypeError, ValueError):
        risk = 0.5
    return max(0.0, min(1.0, base * (1.0 - max(0.0, min(1.0, risk))) * penalty))


def _aggregate(claim_scores: dict):
    if not claim_scores:
        return 0.5, "empty", None
    m = min(claim_scores.values())
    if m < MIN_POOL_THRESHOLD:
        return m, "min", min(claim_scores, key=claim_scores.get)
    C = sum(claim_scores.values()) / len(claim_scores)
    weakest = min(claim_scores, key=claim_scores.get) if len(claim_scores) > 1 else None
    return C, "mean", weakest


class CoIVerbalized(TruthMethod):
    """Chain-of-Interaction verbalized confidence with deterministic aggregation."""

    REQUIRES_SAMPLED_TEXT = False
    REQUIRES_NORMALIZATION = False   # truth_value is already a calibrated-ish [0,1] confidence

    def __init__(self, chain_count: int = 3, max_new_tokens: int = 4096, log_path: str = None,
                 verify_samples: int = None, verify_temperature: float = 0.7,
                 verify_max_tokens: int = 8, **generation_kwargs):
        super().__init__()
        assert chain_count in (1, 2, 3, 4, 5, 6, 7), "chain_count must be 1..7"
        self.chain_count = chain_count
        self.max_new_tokens = max_new_tokens      # long-form multi-claim JSON needs room; models have 40k-131k ctx
        self.log_path = log_path                  # if set, append per-item {raw, phi, truth_value} JSONL
        # decision-framed verify rungs: n=6 = 1 greedy pass; n=7 = K sampled passes (self-consistency)
        if chain_count == 6:
            self.verify_samples, self.verify_do_sample = 1, False
        elif chain_count == 7:
            self.verify_samples, self.verify_do_sample = (verify_samples or 5), True
        else:
            self.verify_samples, self.verify_do_sample = (verify_samples or 1), (verify_samples or 1) > 1
        self.verify_temperature = verify_temperature
        self.verify_max_tokens = verify_max_tokens
        self.generation_kwargs = generation_kwargs

    def _log(self, rec):
        if self.log_path:
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            except OSError:
                pass

    # ---- deterministic phi over parsed chains (shared by hf_local + api) ----
    def _phi(self, obj: dict):
        """Map parsed chains -> {claim_scores, C, pooling, weakest, rationale}. Robust to missing keys."""
        n = self.chain_count
        # claims + per-claim fields
        c1 = (obj.get("chain_1_decompose") or {}).get("claims") or []
        ids = [c.get("id", f"c{i+1}") for i, c in enumerate(c1)] or ["c1"]
        if n == 2:
            conf = {c.get("id", f"c{i+1}"): float(c.get("confidence", 0.5))
                    for i, c in enumerate((obj.get("chain_2_confidence") or {}).get("claims") or [])}
            scores = {cid: max(0.0, min(1.0, conf.get(cid, 0.5))) for cid in ids}
        else:
            if n == 4 or n == 5:
                ev = {c.get("id"): c for c in (obj.get("chain_2a_evidence") or {}).get("claims") or []}
                cr = {c.get("id"): c for c in (obj.get("chain_2b_critique") or {}).get("claims") or []}
                per = {cid: {**ev.get(cid, {}), **cr.get(cid, {})} for cid in ids}
            else:  # n == 3
                per = {c.get("id", f"c{i+1}"): c
                       for i, c in enumerate((obj.get("chain_2_evidence_critique") or {}).get("claims") or [])}
            contr = (obj.get("chain_3_consistency") or {}).get("contradictions") or []
            scores = {cid: _score_claim(per.get(cid, {}).get("grounding"),
                                        per.get(cid, {}).get("risk", 0.5), cid, contr)
                      for cid in ids}
        C, pooling, weakest = _aggregate(scores)
        rat = (f"Aggregate confidence {C:.2f} from a single claim." if weakest is None
               else f"Aggregate confidence {C:.2f} via {pooling} pooling over {len(scores)} claims; weakest {weakest}.")
        return {"claim_confidence": scores, "aggregate_confidence": C,
                "pooling_rule": pooling, "weakest_claim_id": weakest, "rationale": rat}

    def _decompose_claims(self, question, generated_text, gen):
        """One pass -> list of atomic claim texts. Falls back to the whole answer as one claim."""
        raw = gen(_PROMPT_DECOMPOSE.replace("{x}", str(question)).replace("{y}", str(generated_text)),
                  max_new=1024)["generated_text_skip_specials"]
        obj = _extract_json(raw) or {}
        claims = [str(c.get("text", "")).strip() for c in (obj.get("claims") or []) if str(c.get("text", "")).strip()]
        return claims or [str(generated_text).strip()], raw

    def _decision_verify(self, question, generated_text, gen):
        """n=6/7: decompose, then verify each claim with a fresh discriminative YES/NO call
        (n=6: 1 greedy; n=7: K sampled -> vote fraction). phi pools the per-claim YES-fractions."""
        claims, dec_raw = self._decompose_claims(question, generated_text, gen)
        K = self.verify_samples
        scores, per_claim = {}, []
        for i, claim in enumerate(claims):
            cid = f"c{i+1}"
            vp = _PROMPT_VERIFY.replace("{x}", str(question)).replace("{claim}", claim)
            votes = []
            for _ in range(K):
                r = gen(vp, do_sample=self.verify_do_sample,
                        temperature=(self.verify_temperature if self.verify_do_sample else None),
                        max_new=self.verify_max_tokens)["generated_text_skip_specials"]
                votes.append(_parse_yesno(r))
            yes_frac = sum(votes) / len(votes) if votes else 0.5
            scores[cid] = max(0.0, min(1.0, yes_frac))   # P(claim correct); base prior = 1.0 (no grounding axis)
            per_claim.append({"id": cid, "yes_frac": yes_frac, "votes": votes})
        C, pooling, weakest = _aggregate(scores)
        self._log({"q": str(question)[:160], "n": self.chain_count, "K": K, "n_claims": len(claims),
                   "parse_ok": True, "tv": C, "pooling": pooling, "per_claim": per_claim,
                   "dec_raw": dec_raw[:800]})
        return {"truth_value": C, "chain_count": self.chain_count,
                "phi": {"claim_confidence": scores, "aggregate_confidence": C, "pooling_rule": pooling,
                        "weakest_claim_id": weakest, "verify_samples": K}, "coi_raw": dec_raw}

    def _prompt(self, question, generated_text):
        # NOTE: use replace(), not format() -- the templates embed literal JSON braces.
        tmpl = {1: _PROMPT_N1, 2: _PROMPT_N2, 3: _PROMPT_N3, 4: _PROMPT_N4, 5: _PROMPT_N4}[self.chain_count]
        return tmpl.replace("{x}", str(question)).replace("{y}", str(generated_text))

    def forward_hf_local(self, model: PreTrainedModel, input_text: str, generated_text: str,
                         question: str, all_ids: Union[list, torch.Tensor],
                         tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
                         generation_seed=None, sampled_generations_dict: dict = None,
                         messages: list = [], context: str = "", **kwargs):
        def _gen(user, do_sample=False, temperature=None, max_new=None):
            chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            tok, chat = fix_tokenizer_chat(tokenizer, chat)
            # enable_thinking=False: Qwen3 defaults to long chain-of-thought, which inflates latency
            # (the paper's core metric) and can time out; other tokenizers ignore the unknown kwarg.
            try:
                prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                                 continue_final_message=False, enable_thinking=False)
            except TypeError:
                prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                                 continue_final_message=False)
            gk = copy.deepcopy(self.generation_kwargs); gk.pop("do_sample", None)
            sk = {"temperature": temperature, "top_p": 0.95} if (do_sample and temperature) else {}
            return generate(prompt, model, tok, do_sample=do_sample,
                            max_new_tokens=(max_new or self.max_new_tokens), **sk, **gk)

        # ---- decision-framed / self-consistency verify rungs (n=6, n=7) ----
        if self.chain_count in (6, 7):
            return self._decision_verify(question, generated_text, _gen)

        raw = _gen(self._prompt(question, generated_text))["generated_text_skip_specials"]
        parsed = _extract_json(raw)
        obj = parsed or {}

        if self.chain_count == 1:
            m = re.search(r'-?\d+(?:\.\d+)?', str(obj.get("confidence", "")))
            conf = float(m.group()) if m else 50.0
            conf = conf / 100.0 if conf > 1.0 else conf
            tv = max(0.0, min(1.0, conf))
            self._log({"q": question[:160], "n": 1, "raw_len": len(raw), "parse_ok": parsed is not None, "tv": tv})
            return {"truth_value": tv, "coi_raw": raw, "chain_count": 1}

        phi = self._phi(obj)
        self._log({"q": question[:160], "n": self.chain_count, "raw_len": len(raw),
                   "parse_ok": parsed is not None, "n_claims": len(phi["claim_confidence"]),
                   "tv": phi["aggregate_confidence"], "raw": raw[:1500]})
        out = {"truth_value": phi["aggregate_confidence"], "coi_raw": raw,
               "chain_count": self.chain_count, "phi": phi}

        if self.chain_count == 5:  # extra self-rating pass over the full structure
            raw5 = _gen(self._prompt(question, generated_text) +
                        _PROMPT_N5_SUFFIX.format(phi_score=phi["aggregate_confidence"]))["generated_text_skip_specials"]
            o5 = _extract_json(raw5) or {}
            m = re.search(r'-?\d+(?:\.\d+)?', str(o5.get("self_confidence", "")))
            sc = float(m.group()) if m else phi["aggregate_confidence"] * 100
            out["self_confidence"] = max(0.0, min(1.0, sc / 100.0 if sc > 1.0 else sc))
            out["coi_raw_selfrate"] = raw5
            # n=5 reports BOTH; truth_value stays phi (deterministic) by default, self_confidence logged
        return out

    def forward_api(self, model: str, messages: list, generated_text: str, question: str,
                    generation_seed=None, sampled_generations_dict: dict = None, context: str = "", **kwargs):
        from ..generation import sample_generations_api

        def _gen(user):
            chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            d = sample_generations_api(model=model, messages=chat, generation_seed=generation_seed,
                                       number_of_generations=1, temperature=0.0,
                                       max_tokens=self.max_new_tokens)
            return d["generated_texts"][0]

        if self.chain_count in (6, 7):
            def _g(user, do_sample=False, temperature=None, max_new=None):
                chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
                d = sample_generations_api(model=model, messages=chat, generation_seed=generation_seed,
                                           number_of_generations=1,
                                           temperature=(temperature or 0.0) if do_sample else 0.0,
                                           max_tokens=(max_new or self.max_new_tokens))
                return {"generated_text_skip_specials": d["generated_texts"][0]}
            return self._decision_verify(question, generated_text, _g)

        raw = _gen(self._prompt(question, generated_text))
        obj = _extract_json(raw) or {}
        if self.chain_count == 1:
            m = re.search(r'-?\d+(?:\.\d+)?', str(obj.get("confidence", "")))
            conf = float(m.group()) if m else 50.0
            conf = conf / 100.0 if conf > 1.0 else conf
            return {"truth_value": max(0.0, min(1.0, conf)), "coi_raw": raw, "chain_count": 1}
        phi = self._phi(obj)
        out = {"truth_value": phi["aggregate_confidence"], "coi_raw": raw,
               "chain_count": self.chain_count, "phi": phi}
        if self.chain_count == 5:
            raw5 = _gen(self._prompt(question, generated_text) +
                        _PROMPT_N5_SUFFIX.format(phi_score=phi["aggregate_confidence"]))
            o5 = _extract_json(raw5) or {}
            m = re.search(r'-?\d+(?:\.\d+)?', str(o5.get("self_confidence", "")))
            sc = float(m.group()) if m else phi["aggregate_confidence"] * 100
            out["self_confidence"] = max(0.0, min(1.0, sc / 100.0 if sc > 1.0 else sc))
            out["coi_raw_selfrate"] = raw5
        return out
