"""
src/utils.py - Model loading, generation constants, seed management, and factories.
"""
import json
import math
import os
from typing import Any, Callable

# Global random seed specified by course policy
GLOBAL_SEED = 5326

# Sampling & decoding hyperparameter configurations
GREEDY_GEN_KWARGS = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 512,
}

SAMPLING_GEN_KWARGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512,
}

JUDGE_GEN_KWARGS = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 96,
}


def set_seed_for_problem(problem_index: int) -> None:
    """
    Sets a deterministic seed for problem_index (5326 + problem_index).
    Guarantees reproducibility and safe resumability after interruptions.
    """
    try:
        import transformers
        transformers.set_seed(GLOBAL_SEED + problem_index)
    except ImportError:
        import random
        random.seed(GLOBAL_SEED + problem_index)


def load_models(generator_device: str = "cuda:0", judge_device: str = "cuda:1"):
    """
    Loads Qwen2.5-1.5B-Instruct in float16 (Generator) and
    Qwen2.5-7B-Instruct in 4-bit NF4 (Judge).
    
    Hardware Mapping:
      - Kaggle GPU T4x2: generator on cuda:0, judge on cuda:1.
      - Single GPU: both models placed on cuda:0.
      - Warning: Never use device_map="auto" across two GPUs for a single model!
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus < 2:
        g_dev = "cuda:0" if num_gpus == 1 else "cpu"
        j_dev = "cuda:0" if num_gpus == 1 else "cpu"
    else:
        g_dev = generator_device
        j_dev = judge_device

    # 1. Generator: Qwen/Qwen2.5-1.5B-Instruct in float16 (~3.1 GB VRAM)
    gen_id = "Qwen/Qwen2.5-1.5B-Instruct"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_id)
    if gen_tokenizer.pad_token_id is None:
        gen_tokenizer.pad_token_id = gen_tokenizer.eos_token_id

    gen_device_map = {"": g_dev} if g_dev != "cpu" else None
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_id,
        torch_dtype=torch.float16 if g_dev != "cpu" else torch.float32,
        device_map=gen_device_map,
    )
    if g_dev == "cpu":
        gen_model = gen_model.to("cpu")

    # 2. Judge: Qwen/Qwen2.5-7B-Instruct in 4-bit NF4 (~5.5 GB VRAM)
    judge_id = "Qwen/Qwen2.5-7B-Instruct"
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_id)
    if judge_tokenizer.pad_token_id is None:
        judge_tokenizer.pad_token_id = judge_tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    judge_device_map = {"": j_dev} if j_dev != "cpu" else None
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_id,
        quantization_config=bnb_config if j_dev != "cpu" else None,
        torch_dtype=torch.float16 if j_dev != "cpu" else torch.float32,
        device_map=judge_device_map,
    )
    if j_dev == "cpu":
        judge_model = judge_model.to("cpu")

    return gen_model, gen_tokenizer, judge_model, judge_tokenizer


def _get_eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    """Extracts all EOS token IDs from model.generation_config and tokenizer."""
    eos_ids: set[int] = set()
    if hasattr(model, "generation_config") and model.generation_config is not None:
        cfg_eos = getattr(model.generation_config, "eos_token_id", None)
        if cfg_eos is not None:
            if isinstance(cfg_eos, int):
                eos_ids.add(cfg_eos)
            elif isinstance(cfg_eos, (list, tuple, set)):
                eos_ids.update(cfg_eos)
    if tokenizer is not None:
        tok_eos = getattr(tokenizer, "eos_token_id", None)
        if tok_eos is not None:
            if isinstance(tok_eos, int):
                eos_ids.add(tok_eos)
            elif isinstance(tok_eos, (list, tuple, set)):
                eos_ids.update(tok_eos)
    return eos_ids


def _slice_to_first_eos(gen_tokens: Any, eos_ids: set[int]) -> Any:
    """Slices token sequence up to and including the first EOS token."""
    if hasattr(gen_tokens, "tolist"):
        ids = gen_tokens.tolist()
    else:
        ids = list(gen_tokens)
    first_eos_idx = None
    for idx, tok in enumerate(ids):
        if tok in eos_ids:
            first_eos_idx = idx
            break
    if first_eos_idx is not None:
        return gen_tokens[:first_eos_idx + 1]
    return gen_tokens


def make_hf_generate_fn(model: Any, tokenizer: Any) -> Callable[[str, int, dict], list[dict]]:
    """
    Factory creating a GenerateFn conforming to submission_adapter contract.
    Applies the Qwen chat template to raw instruction prompts.
    """
    def generate_fn(prompt: str, num_samples: int, gen_kwargs: dict) -> list[dict]:
        try:
            import torch
            no_grad_ctx = torch.no_grad
        except ImportError:
            class _NoGrad:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            no_grad_ctx = lambda: _NoGrad()

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        kwargs = dict(gen_kwargs)
        if not kwargs.get("do_sample", False):
            kwargs["temperature"] = None
            kwargs["top_p"] = None
            kwargs["top_k"] = None

        if num_samples > 1 and kwargs.get("do_sample", True):
            kwargs["num_return_sequences"] = num_samples

        kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

        with no_grad_ctx():
            output_ids = model.generate(**inputs, **kwargs)

        eos_ids = _get_eos_token_ids(model, tokenizer)
        results = []
        for i in range(output_ids.shape[0]):
            gen_tokens = output_ids[i][prompt_len:]
            gen_tokens_valid = _slice_to_first_eos(gen_tokens, eos_ids)
            text = tokenizer.decode(gen_tokens_valid, skip_special_tokens=True)
            results.append({"text": text, "num_tokens": len(gen_tokens_valid)})
        return results

    return generate_fn


def make_hf_judge_fn(judge_model: Any, judge_tokenizer: Any) -> Callable[[str], dict]:
    """
    Factory creating a JudgeFn conforming to submission_adapter contract.
    Applies the Qwen chat template to raw judge prompts.
    """
    def judge_fn(judge_prompt: str) -> dict:
        try:
            import torch
            no_grad_ctx = torch.no_grad
        except ImportError:
            class _NoGrad:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            no_grad_ctx = lambda: _NoGrad()

        messages = [{"role": "user", "content": judge_prompt}]
        formatted = judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = judge_tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(judge_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        kwargs = dict(JUDGE_GEN_KWARGS)
        kwargs.setdefault("pad_token_id", judge_tokenizer.eos_token_id)

        with no_grad_ctx():
            output_ids = judge_model.generate(**inputs, **kwargs)

        eos_ids = _get_eos_token_ids(judge_model, judge_tokenizer)
        gen_tokens = output_ids[0][prompt_len:]
        gen_tokens_valid = _slice_to_first_eos(gen_tokens, eos_ids)
        text = judge_tokenizer.decode(gen_tokens_valid, skip_special_tokens=True)
        return {"text": text, "num_tokens": len(gen_tokens_valid)}

    return judge_fn


def save_results_summary(results_list: list[dict[str, Any]], filepath: str = "outputs/results_summary.json") -> None:
    """
    Saves results_summary.json conforming strictly to the assignment schema:
    Keys: 'Method', 'Accuracy (%)', 'Generation Calls', 'Judge Calls', 'Total LLM Calls',
          'Avg Gen Tokens / Problem', 'Avg Judge Tokens / Problem', 'Total Time (s)'
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)


def save_aggregation_ablation(ablation_list: list[dict[str, Any]], filepath: str = "outputs/aggregation_ablation.json") -> None:
    """
    Saves aggregation_ablation.json:
    List of {'Strategy': str, 'Lambda': float | None, 'Accuracy (%)': float, 'Tie Rate (%)': float}
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(ablation_list, f, indent=2)


def save_sc_scaling_curve(curve_list: list[dict[str, Any]], filepath: str = "outputs/sc_scaling_curve.json") -> None:
    """
    Saves sc_scaling_curve.json:
    List of {'N': int, 'Accuracy (%)': float, 'Avg Gen Tokens / Problem': float}
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(curve_list, f, indent=2)


# ===========================================================================
# Additions (student implementation)
# ===========================================================================
def make_hf_batched_judge_fn(judge_model: Any, judge_tokenizer: Any, batch_size: int = 8):
    """
    Batched version of the judge: takes a LIST of raw judge prompts and returns a list
    of {"text", "num_tokens"} dicts in the same order.

    Why: the 7B judge is the bottleneck (~1,250 step verdicts for the full benchmark).
    Batching 8 prompts per .generate() call cuts judging time several-fold on a T4.
    Uses LEFT padding so every prompt ends exactly where generation starts; tokens are
    counted from IDs after the (padded) prompt and sliced at the first EOS, exactly
    like make_hf_judge_fn.

    In the pipeline the outputs are put into a dict {prompt: output}, and score_path is
    called with a judge_fn that looks prompts up in that dict, so the graded code path
    (score_path -> build_prm_judge_prompt -> judge_fn -> parse_judge_decision) is unchanged.
    """
    import torch

    eos_ids = _get_eos_token_ids(judge_model, judge_tokenizer)

    def batched_judge_fn(prompts: list[str]) -> list[dict]:
        if not prompts:
            return []
        formatted = [
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        # Sort by length to minimise padding, then restore order.
        order = sorted(range(len(formatted)), key=lambda i: len(formatted[i]))
        results: list[dict | None] = [None] * len(formatted)

        old_side = judge_tokenizer.padding_side
        judge_tokenizer.padding_side = "left"
        try:
            for b in range(0, len(order), batch_size):
                idx = order[b:b + batch_size]
                enc = judge_tokenizer([formatted[i] for i in idx], return_tensors="pt", padding=True)
                enc = {k: v.to(judge_model.device) for k, v in enc.items()}
                prompt_len = enc["input_ids"].shape[1]
                kwargs = dict(JUDGE_GEN_KWARGS)
                kwargs.setdefault("pad_token_id", judge_tokenizer.pad_token_id or judge_tokenizer.eos_token_id)
                with torch.no_grad():
                    out = judge_model.generate(**enc, **kwargs)
                for row, i in enumerate(idx):
                    gen_tokens = _slice_to_first_eos(out[row][prompt_len:], eos_ids)
                    text = judge_tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    results[i] = {"text": text, "num_tokens": len(gen_tokens)}
        finally:
            judge_tokenizer.padding_side = old_side
        return results  # type: ignore[return-value]

    return batched_judge_fn


def make_verdict_prob_fn(judge_model: Any, judge_tokenizer: Any, batch_size: int = 8):
    """
    OPTIONAL BONUS (+3): continuous step confidence from the judge's logits.

    For each (judge_prompt, judge_text) pair we take the judge's own generated text up to
    and including its first "Verdict:" (or append "\\nVerdict:" if it never wrote one) and
    compute the FULL-SEQUENCE log-probability of the two continuations " VALID" and
    " INVALID" (summing log-probs over all of each continuation's tokens, so it does not
    matter that the two words tokenize into different numbers of pieces). Then

        p_valid = exp(lp_VALID) / (exp(lp_VALID) + exp(lp_INVALID))

    State in the report that the full-sequence approach (not first-token) was used.
    Returns fn(list[(prompt, judge_text)]) -> list[float].
    """
    import re as _re
    import torch

    verdict_re = _re.compile(r"verdict\s*:", _re.IGNORECASE)
    conts = [" VALID", " INVALID"]

    def _prefix(prompt: str, judge_text: str) -> str:
        chat = judge_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        m = verdict_re.search(judge_text or "")
        if m:
            assistant = judge_text[: m.end()]
        else:
            assistant = (judge_text or "").rstrip() + "\nVerdict:"
        return chat + assistant

    def prob_fn(items: list[tuple[str, str]]) -> list[float]:
        # Build 2 sequences per item: prefix+" VALID", prefix+" INVALID".
        seqs, spans = [], []
        for prompt, text in items:
            pre = _prefix(prompt, text)
            pre_ids = judge_tokenizer(pre, add_special_tokens=False)["input_ids"]
            for c in conts:
                full_ids = judge_tokenizer(pre + c, add_special_tokens=False)["input_ids"]
                # continuation tokens = everything after the shared prefix
                k = 0
                while k < min(len(pre_ids), len(full_ids)) and pre_ids[k] == full_ids[k]:
                    k += 1
                seqs.append(full_ids)
                spans.append(k)

        logps: list[float] = []
        pad_id = judge_tokenizer.pad_token_id or judge_tokenizer.eos_token_id
        for b in range(0, len(seqs), 2 * batch_size):
            chunk = seqs[b:b + 2 * batch_size]
            chunk_spans = spans[b:b + 2 * batch_size]
            L = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), L), dtype=torch.long)
            for r, s in enumerate(chunk):  # right padding: positions stay aligned
                ids[r, : len(s)] = torch.tensor(s)
                mask[r, : len(s)] = 1
            ids, mask = ids.to(judge_model.device), mask.to(judge_model.device)
            # Run the transformer body only, then apply lm_head at the handful of positions
            # we need. Full logits would be (batch x seq_len x 152k vocab) and OOM a T4.
            base = getattr(judge_model, "model", None) or judge_model.get_decoder()
            lm_head = judge_model.get_output_embeddings()
            with torch.no_grad():
                hidden = base(input_ids=ids, attention_mask=mask).last_hidden_state
                for r, s in enumerate(chunk):
                    start = max(1, chunk_spans[r])
                    positions = list(range(start - 1, len(s) - 1))  # token at p+1 predicted at p
                    if not positions:
                        logps.append(0.0)
                        continue
                    h = hidden[r, positions].to(lm_head.weight.dtype)
                    lp = torch.log_softmax(lm_head(h).float(), dim=-1)
                    targets = torch.tensor(s[start:], device=lp.device)
                    logps.append(lp.gather(1, targets[:, None]).sum().item())

        probs = []
        for i in range(0, len(logps), 2):
            lv, li = logps[i], logps[i + 1]
            m = max(lv, li)
            ev, ei = math.exp(lv - m), math.exp(li - m)
            probs.append(ev / (ev + ei))
        return probs

    return prob_fn


def write_jsonl_line(path: str, record: dict) -> None:
    """Appends one JSON record and flushes (checkpointing for Kaggle disconnects)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a half-written last line after a crash is skipped
    return rows