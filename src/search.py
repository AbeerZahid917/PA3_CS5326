"""
src/search.py - Path aggregation, Best-of-N search, and compute-matched search.
"""
import time
from typing import Any, Callable

from .generation import (
    _resolve_generate_fn,
    compute_majority_vote,
    sample_candidates,
)
from .prm_judge import _resolve_judge_fn, score_path

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]

TIE_TOLERANCE = 1e-9
VALID_STRATEGIES = ("min", "product", "step_penalty", "product_logprob")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def aggregate_path_score(
    step_scores: list[float],
    strategy: str = "min",
    penalty_lambda: float = 2.0
) -> float:
    """
    Aggregates a list of step scores (+1.0 or -1.0) into a scalar trajectory score.

    Rules:
      - An empty list returns -1.0 for every strategy.
      - 'min': min(step_scores).
      - 'product': p_t = clamp((r_t + 1)/2, 0.01, 0.99); returns the raw product.
      - 'step_penalty': mean(step_scores) - penalty_lambda * count(r_t == -1.0).
      - An unknown strategy raises ValueError.

    Optional bonus:
      - 'product_logprob': step_scores are CONTINUOUS probabilities p_t = P(VALID)
        computed from the judge's logits; returns prod_t clamp(p_t, 1e-6, 1 - 1e-6).
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown aggregation strategy: {strategy!r}")

    scores = [float(s) for s in (step_scores or [])]
    if not scores:
        return -1.0

    if strategy == "min":
        return min(scores)

    if strategy == "product":
        prod = 1.0
        for r in scores:
            prod *= _clamp((r + 1.0) / 2.0, 0.01, 0.99)
        return prod

    if strategy == "step_penalty":
        n_invalid = sum(1 for r in scores if r == -1.0)
        return sum(scores) / len(scores) - float(penalty_lambda) * n_invalid

    # product_logprob (bonus)
    prod = 1.0
    for p in scores:
        prod *= _clamp(p, 1e-6, 1.0 - 1e-6)
    return prod


def _scores_for(cand: dict[str, Any], strategy: str) -> list[float]:
    """The bonus strategy ranks with continuous probabilities stored in 'step_probs'."""
    if strategy == "product_logprob":
        probs = cand.get("step_probs")
        if probs is None:
            raise ValueError("product_logprob needs candidates with 'step_probs'.")
        return probs
    return cand.get("step_scores") or []


def select_best_path(
    candidate_paths: list[dict[str, Any]],
    aggregation_strategy: str = "min",
    penalty_lambda: float = 2.0
) -> dict[str, Any]:
    """
    Selects the winning candidate path from candidate_paths.

    Input Dictionary Contract:
      Each candidate dictionary in candidate_paths must contain 'step_scores' (list[float])
      and 'answer' (int | None). Any other keys present in the candidate dictionary
      are passed through unchanged in the returned copy.

    Selection & Tie-Breaking Rules:
      1. Compute path_score for each candidate with aggregate_path_score.
         A candidate with an empty step_scores list gets -1.0.
      2. Eligibility: if at least one candidate has a non-None answer, candidates
         with answer=None are excluded. If all answers are None, all remain eligible.
      3. Find maximum path_score among eligible candidates (tolerance 1e-9).
         The tied set is all eligible candidates within tolerance of the maximum.
      4. If the tied set has more than one candidate, compute compute_majority_vote
         over their answers. Return the lowest-index candidate in the tied set whose
         answer equals the vote. If the vote is None, return the lowest-index tied candidate.
      5. Returns a shallow COPY of the chosen dict with two added keys:
         'path_score' (float) and 'num_tied' (int, size of the tied set).
         Does not mutate the inputs.
    """
    if not candidate_paths:
        raise ValueError("select_best_path needs at least one candidate.")
    if aggregation_strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy!r}")

    # 1. path scores (aggregate_path_score already maps [] -> -1.0)
    path_scores = [
        aggregate_path_score(_scores_for(c, aggregation_strategy), aggregation_strategy, penalty_lambda)
        for c in candidate_paths
    ]

    # 2. eligibility (filter answer=None when any real answer exists)
    indices = list(range(len(candidate_paths)))
    if any(c.get("answer") is not None for c in candidate_paths):
        eligible = [i for i in indices if candidate_paths[i].get("answer") is not None]
    else:
        eligible = indices

    # 3. tied set at the maximum score
    best_score = max(path_scores[i] for i in eligible)
    tied = [i for i in eligible if abs(path_scores[i] - best_score) <= TIE_TOLERANCE]

    # 4. tie-break by majority vote among tied candidates
    chosen = tied[0]
    if len(tied) > 1:
        vote = compute_majority_vote([candidate_paths[i].get("answer") for i in tied])
        if vote is not None:
            chosen = next(i for i in tied if candidate_paths[i].get("answer") == vote)

    # 5. shallow copy with extra keys
    result = dict(candidate_paths[chosen])
    result["path_score"] = float(path_scores[chosen])
    result["num_tied"] = len(tied)
    return result


def run_prm_best_of_n(
    model: Any = None,
    judge_model: Any = None,
    gen_tokenizer: Any = None,
    judge_tokenizer: Any = None,
    question: str = "",
    n: int = 5,
    aggregation_strategy: str = "min",
    penalty_lambda: float = 2.0,
    candidates: list[dict[str, Any]] | None = None,
    generate_fn: GenerateFn | None = None,
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Executes PRM-guided Best-of-N search over reasoning trajectories.

    If candidates is provided (shared candidate pool), does NOT sample new candidates
    (generation_calls = 0, generated_tokens = 0). Provided candidates use the
    run_self_consistency candidate format (at minimum 'steps' and 'answer').
    Otherwise, samples n candidates using run_self_consistency.

    Scores each candidate's steps using score_path.
    Selects the winning candidate via select_best_path.

    Returns:
      selected, answer, candidates [each augmented with step_scores, rationales],
      generation_calls, judge_calls, generated_tokens, judge_tokens, time_seconds
    """
    start = time.perf_counter()

    generation_calls = 0
    generated_tokens = 0
    if candidates is None:
        gen = _resolve_generate_fn(model, gen_tokenizer, generate_fn)
        pool, _ = sample_candidates(gen, question, n)
        generation_calls = len(pool)
        generated_tokens = sum(c["num_tokens"] for c in pool)
    else:
        pool = list(candidates)

    judge = _resolve_judge_fn(judge_model, judge_tokenizer, judge_fn) if pool else None

    scored: list[dict[str, Any]] = []
    judge_calls = 0
    judge_tokens = 0
    for cand in pool:
        sp = score_path(question=question, candidate=cand, judge_fn=judge)
        aug = dict(cand)  # do not mutate caller's dicts
        aug["step_scores"] = sp["step_scores"]
        aug["rationales"] = sp["rationales"]
        aug["judge_outputs"] = sp.get("judge_outputs", [])
        aug["judge_tokens"] = sp["judge_tokens"]
        scored.append(aug)
        judge_calls += sp["judge_calls"]
        judge_tokens += sp["judge_tokens"]

    selected = select_best_path(scored, aggregation_strategy, penalty_lambda) if scored else None

    return {
        "selected": selected,
        "answer": selected.get("answer") if selected else None,
        "candidates": scored,
        "generation_calls": generation_calls,
        "judge_calls": judge_calls,
        "generated_tokens": generated_tokens,
        "judge_tokens": judge_tokens,
        "time_seconds": time.perf_counter() - start,
    }


def run_compute_matched_self_consistency(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    target_calls: int = 15,
    temperature: float = 0.7,
    existing_candidates: list[dict[str, Any]] | None = None,
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes compute-matched Self-Consistency aggregating target_calls paths.
    If existing_candidates are supplied, samples only the deficit:
    target_calls - len(existing_candidates). Returned candidates = existing_candidates followed by new samples, in sampling order.

    Returns:
      candidates, answer, generation_calls (new samples only), total_samples,
      generated_tokens (new samples only), time_seconds
    """
    start = time.perf_counter()
    existing = list(existing_candidates or [])
    deficit = max(0, int(target_calls) - len(existing))

    new: list[dict[str, Any]] = []
    if deficit > 0:
        gen = _resolve_generate_fn(model, tokenizer, generate_fn)
        new, _ = sample_candidates(gen, question, deficit, temperature)

    all_cands = existing + new
    answer = compute_majority_vote([c.get("answer") for c in all_cands])
    return {
        "candidates": all_cands,
        "answer": answer,
        "generation_calls": len(new),
        "total_samples": len(all_cands),
        "generated_tokens": sum(int(c.get("num_tokens", 0) or 0) for c in new),
        "time_seconds": time.perf_counter() - start,
    }