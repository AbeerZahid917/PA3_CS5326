"""
src/generation.py - Prompt formatting and baseline generation routines.

All model access goes through a GenerateFn:
    generate_fn(raw_prompt, num_samples, gen_kwargs) -> [{"text": str, "num_tokens": int}, ...]
On Kaggle this is built by src.utils.make_hf_generate_fn (which applies the Qwen chat
template and counts tokens from token IDs). In the autograder it is a mock.
"""
import time
from collections import Counter
from typing import Any, Callable

from .extraction import extract_answer, parse_steps

GenerateFn = Callable[[str, int, dict], list[dict]]
# generate_fn(raw_prompt, num_samples, gen_kwargs) -> list of {"text": str, "num_tokens": int}
# gen_kwargs carries do_sample/temperature/top_p/top_k/repetition_penalty/max_new_tokens

# Decoding settings (mirrors src/utils.py constants; duplicated here so this module
# can be imported without torch/transformers).
GREEDY_KWARGS = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 512,
}

SAMPLING_KWARGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512,
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_STEP_PROMPT_TEMPLATE = """Solve the following grade-school math word problem by reasoning step by step.

Question: {question}

Follow these rules exactly:
1. Write your reasoning as numbered steps. Start each step on a new line with "Step 1:", "Step 2:", "Step 3:", and so on.
2. Each step must contain one deduction or one calculation. Write every calculation out in full, for example "3 + 4 = 7".
3. Use plain text only. Do not use markdown, headings, bold text, bullet points, LaTeX, or \\boxed{{}}.
4. After the last step, write one final line in exactly this form:
Final Answer: <integer>
5. The final answer must be a single whole number with no units, no dollar sign, and no commas. Write nothing after the Final Answer line.

Your response must look like this:
Step 1: <first deduction or calculation>
Step 2: <next deduction or calculation>
...
Step K: <final calculation>
Final Answer: <integer>"""


def format_step_prompt(question: str) -> str:
    """
    Formats the raw problem question into a prompt that instructs the
    model to emit numbered steps ('Step 1: ...') and 'Final Answer: <int>'.

    Returns a raw instruction string (no chat template applied).
    """
    question = (question or "").strip()
    return _STEP_PROMPT_TEMPLATE.format(question=question)


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------
def compute_majority_vote(answers: list[int | None]) -> int | None:
    """
    Given a list of extracted answers, returns the majority answer.
    Ignores None values. Breaks ties deterministically in favor of the answer
    that appeared earliest in the list.
    """
    valid = [a for a in (answers or []) if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    best = max(counts.values())
    # Walk in original order: the first answer reaching the max count wins ties.
    for a in valid:
        if counts[a] == best:
            return a
    return None  # unreachable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_generate_fn(model: Any, tokenizer: Any, generate_fn: GenerateFn | None) -> GenerateFn:
    if generate_fn is not None:
        return generate_fn
    if model is None or tokenizer is None:
        raise ValueError("Provide either generate_fn or both model and tokenizer.")
    from .utils import make_hf_generate_fn  # lazy: avoids importing torch in tests
    return make_hf_generate_fn(model, tokenizer)


def make_candidate(output: dict) -> dict[str, Any]:
    """Turns one generate_fn output into the standard candidate dict."""
    text = output.get("text", "") or ""
    return {
        "completion": text,
        "steps": parse_steps(text),
        "answer": extract_answer(text),
        "num_tokens": int(output.get("num_tokens", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def run_greedy(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes greedy decoding for a single question.
    Calls generate_fn with num_samples=1 and gen_kwargs containing do_sample=False,
    temperature=None, top_p=None, top_k=None.

    Returns:
      prompt, completion, steps, answer, generation_calls, generated_tokens, time_seconds
    """
    gen = _resolve_generate_fn(model, tokenizer, generate_fn)
    prompt = format_step_prompt(question)

    start = time.perf_counter()
    outputs = gen(prompt, 1, dict(GREEDY_KWARGS))
    elapsed = time.perf_counter() - start

    out = outputs[0] if outputs else {"text": "", "num_tokens": 0}
    cand = make_candidate(out)
    return {
        "prompt": prompt,
        "completion": cand["completion"],
        "steps": cand["steps"],
        "answer": cand["answer"],
        "generation_calls": 1,
        "generated_tokens": cand["num_tokens"],
        "time_seconds": elapsed,
    }


def sample_candidates(
    gen: GenerateFn, question: str, n: int, temperature: float = 0.7
) -> tuple[list[dict[str, Any]], float]:
    """Samples n candidates in one batched call. Returns (candidates, seconds)."""
    if n <= 0:
        return [], 0.0
    prompt = format_step_prompt(question)
    kwargs = dict(SAMPLING_KWARGS)
    kwargs["temperature"] = temperature
    start = time.perf_counter()
    outputs = gen(prompt, n, kwargs)
    elapsed = time.perf_counter() - start
    return [make_candidate(o) for o in outputs[:n]], elapsed


def run_self_consistency(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    n: int = 5,
    temperature: float = 0.7,
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Samples n diverse reasoning trajectories and aggregates candidate answers via majority vote.

    Returns:
      candidates [each: completion, steps, answer, num_tokens], answer,
      generation_calls, generated_tokens, time_seconds
    """
    gen = _resolve_generate_fn(model, tokenizer, generate_fn)
    start = time.perf_counter()
    candidates, _ = sample_candidates(gen, question, n, temperature)
    answer = compute_majority_vote([c["answer"] for c in candidates])
    elapsed = time.perf_counter() - start
    return {
        "candidates": candidates,
        "answer": answer,
        "generation_calls": len(candidates),  # one sequence = one call
        "generated_tokens": sum(c["num_tokens"] for c in candidates),
        "time_seconds": elapsed,
    }