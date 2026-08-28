#!/usr/bin/env python
"""Faithful controlled DALD proxy — a third proxy baseline for the all-methods comparison.

DALD (Zeng et al., NeurIPS 2024) trains a small white-box surrogate by **LoRA SFT on a black-box
model's (prompt, response) pairs with the prompt masked** (loss on response tokens only), so
logit-based scoring can be applied. This reimplements exactly that objective on OUR footing —
same base (Qwen3-0.6B), same teacher data (sft_qwen3-8b), saved in the structure rq5_score.py
reads — so DALD, DisAAD, and our uncertainty-aware method are compared with only the *training
objective* differing:

    DALD  = LoRA SFT, prompt-masked (response-only loss)          <-- this file
    DisAAD= LoRA SFT (whole-sequence) + adversarial
    Ours  = LoRA SFT + adversarial + uncertainty-aware term

Mirrors DALD's train_dald.py (HF Trainer + DataCollatorForSeq2Seq + LoRA), not DisAAD's adversarial
loop. Runs in the `disaad` venv (transformers 4.56 / peft).

    python scripts/dald_train.py
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description="Faithful controlled DALD proxy (masked LoRA SFT).")
    ap.add_argument("--sft-data", default=os.path.expanduser("~/JasonLucas/outputs/disaad/sft_qwen3-8b.raw_data.json"))
    ap.add_argument("--student", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--student-key", default=None,
                    help="dir name for the saved adapter (default: derived from --student). "
                         "rq5_score.py globs <out>/<student-key>/logs/saved_models/best_model.")
    ap.add_argument("--out", default=os.path.expanduser("~/JasonLucas/outputs/disaad/proxy_dald"))
    ap.add_argument("--teacher", default="qwen3-8b", help="teacher tag, recorded in the manifest")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--cutoff-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/JasonLucas/hf_cache"))
    args = ap.parse_args()
    if args.student_key is None:
        args.student_key = args.student.rstrip("/").split("/")[-1].lower()

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                              DataCollatorForSeq2Seq)
    from peft import LoraConfig, get_peft_model

    d = json.load(open(args.sft_data))
    # Match DisAAD's task_loss data EXACTLY: it flattens over blackbox_answers = sft_text[i][1:]
    # (one training pair per teacher sample). DALD differs ONLY by masking the prompt + no adversarial.
    prompts, responses = [], []
    for p, texts in zip(d["sft_prompt"], d["sft_text"]):
        answers = texts[1:] if isinstance(texts, list) else [texts]
        for a in answers:
            if a and a.strip():
                prompts.append(p.strip()); responses.append(a.strip())
    print(f"[dald] {len(prompts)} (prompt,response) pairs from {len(d['sft_prompt'])} prompts; student={args.student}")

    tok = AutoTokenizer.from_pretrained(args.student, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(ex):
        # DALD: mask the prompt, loss on response tokens only
        p_ids = tok(ex["prompt"], add_special_tokens=True)["input_ids"]
        full = tok(ex["prompt"] + ex["response"], truncation=True, max_length=args.cutoff_len,
                   add_special_tokens=True)
        plen = min(len(p_ids), len(full["input_ids"]))
        labels = [-100] * plen + full["input_ids"][plen:]
        full["labels"] = labels
        return full

    ds = Dataset.from_dict({"prompt": prompts, "response": responses})
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16,
                                                 local_files_only=True, trust_remote_code=True)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # save into the structure rq5_score.py globs: <out>/<student-key>/logs/saved_models/best_model
    save_dir = os.path.join(args.out, args.student_key, "logs", "saved_models", "best_model")
    ta = TrainingArguments(
        output_dir=os.path.join(args.out, args.student_key, "logs", "hf_trainer"),
        remove_unused_columns=False, save_strategy="no", learning_rate=args.lr,
        per_device_train_batch_size=1, gradient_accumulation_steps=4, bf16=True,
        num_train_epochs=args.epochs, logging_steps=10, report_to=[])
    trainer = Trainer(model=model, args=ta, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)                       # LoRA adapter
    json.dump({"ready": True, "method": "DALD", "objective": "masked LoRA SFT on teacher outputs",
               "student": args.student, "student_key": args.student_key, "teacher": args.teacher,
               "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "n_pairs": len(prompts)},
              open(os.path.join(args.out, "disaad_ready.json"), "w"), indent=2)
    print(f"[dald] adapter -> {save_dir}")


if __name__ == "__main__":
    main()
