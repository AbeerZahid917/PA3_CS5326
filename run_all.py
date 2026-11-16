"""
run_all.py - End-to-end PA3 evaluation pipeline (runnable top to bottom).

Per problem (seed = 5326 + problem_index, set once before the problem):
  1. Greedy (N=1)
  2. Sample 5 shared candidates                       -> Self-Consistency (N=5)
  3. Sample 10 more (indices 5..14)                     -> Compute-Matched SC (N=15)
  4. Judge every step of the 5 shared candidates ONCE  -> PRM Best-of-5 (min / product / step_penalty)
  5. (bonus) continuous P(VALID) per step from judge logits -> product_logprob
  6. Append one line to outputs/per_problem.jsonl (checkpoint; reruns skip finished problems)
Finally, all metrics are recomputed from per_problem.jsonl and written to
  outputs/results_summary.json, outputs/aggregation_ablation.json, outputs/sc_scaling_curve.json
  (+ outputs/bonus_logprob.json if the bonus ran).

Usage (from the repository root):
  python run_all.py                      # full run on Kaggle T4x2 (resumes automatically)
  python run_all.py --check-format       # greedy-only delimiter-compliance check on all 50
  python run_all.py --limit 3            # quick trial on the first 3 problems
  python run_all.py --summarize-only     # rebuild the JSON artifacts from per_problem.jsonl
  python run_all.py --mock --out-dir outputs_mock   # offline dry run with fake models (no GPU)
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extraction import extract_answer
from src.generation import compute_majority_vote, run_greedy, run_self_consistency
from src.prm_judge import build_prm_judge_prompt
from src.search import (
    run_compute_matched_self_consistency,
    run_prm_best_of_n,
    select_best_path,
)
from src.utils import (
    read_jsonl,
    save_aggregation_ablation,
    save_results_summary,
    save_sc_scaling_curve,
    set_seed_for_problem,
    write_jsonl_line,
)

N_SHARED = 5
N_MATCHED = 15

# Method names exactly as in the handout table.
M_GREEDY = "Greedy (N=1)"
M_SC5 = "Self-Consistency (N=5)"
M_SC15 = "Compute-Matched SC (N=15)"
M_PRM_MIN = "PRM Best-of-5 (min)"
M_PRM_PROD = "PRM Best-of-5 (product)"
M_PRM_PEN = "PRM Best-of-5 (step_penalty, lambda=2.0)"

# (selection key, strategy, lambda) -- every PRM variant reuses the same step scores.
SELECTIONS = [
    ("min", "min", 2.0),
    ("product", "product", 2.0),
    ("step_penalty@0.0", "step_penalty", 0.0),
    ("step_penalty@0.5", "step_penalty", 0.5),
    ("step_penalty@2.0", "step_penalty", 2.0),
]
BONUS_KEY = "product_logprob"


# ---------------------------------------------------------------------------
# Mock models (offline dry run only)
# ---------------------------------------------------------------------------
def make_mock_fns(problems):
    gt_by_q = {p["question"].strip(): p["ground_truth_answer"] for p in problems}

    def mock_generate_fn(prompt, num_samples, gen_kwargs):
        m = re.search(r"Question: (.*?)\n\nFollow these rules", prompt, re.S)
        gt = gt_by_q.get(m.group(1).strip(), 0) if m else 0
        outs = []
        for _ in range(num_samples):
            k = random.randint(2, 6)
            if gen_kwargs.get("do_sample"):
                ans = gt if random.random() < 0.65 else gt + random.choice([-10, -1, 1, 2, 5])
            else:
                ans = gt if hash(prompt) % 3 else gt + 1
            steps = "\n".join(f"Step {i + 1}: compute {i} + 1 = {i + 1}." for i in range(k))
            tail = f"\nFinal Answer: {ans}" if random.random() > 0.05 else "\n\\boxed{%d}" % ans
            outs.append({"text": steps + tail, "num_tokens": 20 * k + random.randint(0, 15)})
        return outs

    def mock_judge_one(prompt):
        h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        verdict = "VALID" if h % 5 else "INVALID"
        return {"text": f"Rationale: The arithmetic checks out as {h % 97}.5 units.\nVerdict: {verdict}",
                "num_tokens": 18 + h % 7}

    def mock_batched_judge(prompts):
        return [mock_judge_one(p) for p in prompts]

    def mock_prob_fn(items):
        out = []
        for _, text in items:
            base = 0.9 if "Verdict: VALID" in text else 0.15
            out.append(min(0.999, max(0.001, base + random.uniform(-0.08, 0.08))))
        return out

    return mock_generate_fn, mock_judge_one, mock_batched_judge, mock_prob_fn


# ---------------------------------------------------------------------------
# One problem
# ---------------------------------------------------------------------------
def run_problem(idx, prob, gen_fn, judge_one_fn, batched_judge_fn, prob_fn):
    q = prob["question"]
    set_seed_for_problem(idx)

    # 1. Greedy
    greedy = run_greedy(question=q, generate_fn=gen_fn)

    # 2. Shared pool of 5 samples (SC@5)
    sc5 = run_self_consistency(question=q, n=N_SHARED, temperature=0.7, generate_fn=gen_fn)

    # 3. +10 samples (SC@15), indices 0-4 are the shared pool
    sc15 = run_compute_matched_self_consistency(
        question=q, target_calls=N_MATCHED, temperature=0.7,
        existing_candidates=sc5["candidates"], generate_fn=gen_fn,
    )

    # 4. Judge every step of the 5 shared candidates once (batched), then run the graded
    #    run_prm_best_of_n path with a judge_fn that reads from that cache.
    t0 = time.perf_counter()
    prompts = []
    for c in sc5["candidates"]:
        for t, step in enumerate(c["steps"]):
            prompts.append(build_prm_judge_prompt(q, c["steps"][:t], step))
    unique_prompts = list(dict.fromkeys(prompts))
    cache = dict(zip(unique_prompts, batched_judge_fn(unique_prompts))) if unique_prompts else {}
    batch_time = time.perf_counter() - t0

    def cached_judge_fn(p):
        return cache[p] if p in cache else judge_one_fn(p)

    prm = run_prm_best_of_n(
        question=q, n=N_SHARED, aggregation_strategy="min",
        candidates=sc5["candidates"], judge_fn=cached_judge_fn,
    )
    judging_time = batch_time + prm["time_seconds"]
    scored = prm["candidates"]
    for i, c in enumerate(scored):
        c["cand_index"] = i

    # 5. Bonus: continuous P(VALID) per step
    bonus_time = None
    if prob_fn is not None:
        try:
            t1 = time.perf_counter()
            items, owners = [], []
            for i, c in enumerate(scored):
                for t, step in enumerate(c["steps"]):
                    items.append((build_prm_judge_prompt(q, c["steps"][:t], step), c["judge_outputs"][t]))
                    owners.append(i)
            probs = prob_fn(items) if items else []
            for c in scored:
                c["step_probs"] = []
            for i, p in zip(owners, probs):
                scored[i]["step_probs"].append(float(p))
            bonus_time = time.perf_counter() - t1
        except Exception as e:  # never let the bonus kill the main run
            print(f"  [bonus] logprob scoring failed on {prob['id']}: {e}")
            for c in scored:
                c.pop("step_probs", None)

    # Selection timing per strategy (each PRM row's "own aggregation/selection time")
    select_times = {}
    for key, strat, lam in SELECTIONS:
        t2 = time.perf_counter()
        select_best_path(scored, strat, lam)
        select_times[key] = time.perf_counter() - t2

    # 6. Record
    candidates = []
    for i, c in enumerate(sc15["candidates"]):
        rec = {k: c[k] for k in ("completion", "steps", "answer", "num_tokens")}
        if i < N_SHARED:
            s = scored[i]
            rec.update({
                "step_scores": s["step_scores"],
                "rationales": s["rationales"],
                "judge_outputs": s["judge_outputs"],
                "judge_tokens": s["judge_tokens"],
            })
            if "step_probs" in s:
                rec["step_probs"] = s["step_probs"]
        candidates.append(rec)

    record = {
        "index": idx,
        "id": prob["id"],
        "question": q,
        "ground_truth_answer": prob["ground_truth_answer"],
        "greedy": {
            "completion": greedy["completion"],
            "steps": greedy["steps"],
            "answer": greedy["answer"],
            "num_tokens": greedy["generated_tokens"],
            "time_seconds": greedy["time_seconds"],
        },
        "candidates": candidates,
        "timing": {
            "greedy_s": greedy["time_seconds"],
            "sc5_sampling_s": sc5["time_seconds"],
            "sc15_extra_sampling_s": sc15["time_seconds"],
            "judging_s": judging_time,
            "select_s": select_times,
            "bonus_logprob_s": bonus_time,
        },
        "prm": {
            "judge_calls": prm["judge_calls"],
            "judge_tokens": prm["judge_tokens"],
            "unique_judge_prompts": len(unique_prompts),
        },
    }
    record["predictions"] = predictions_for(record)
    record["correct"] = {m: (a is not None and a == record["ground_truth_answer"])
                         for m, a in record["predictions"].items()}
    return record


# ---------------------------------------------------------------------------
# Metrics (always recomputed from per_problem records)
# ---------------------------------------------------------------------------
def selections_for(record):
    cands = [dict(c, cand_index=i) for i, c in enumerate(record["candidates"][:N_SHARED])]
    out = {}
    for key, strat, lam in SELECTIONS:
        s = select_best_path(cands, strat, lam)
        out[key] = {"index": s["cand_index"], "answer": s["answer"],
                    "num_tied": s["num_tied"], "path_score": s["path_score"]}
    if all("step_probs" in c for c in cands):
        s = select_best_path(cands, BONUS_KEY)
        out[BONUS_KEY] = {"index": s["cand_index"], "answer": s["answer"],
                          "num_tied": s["num_tied"], "path_score": s["path_score"]}
    return out


def predictions_for(record):
    answers = [c["answer"] for c in record["candidates"]]
    sel = selections_for(record)
    preds = {
        M_GREEDY: record["greedy"]["answer"],
        M_SC5: compute_majority_vote(answers[:N_SHARED]),
        M_SC15: compute_majority_vote(answers[:N_MATCHED]),
        M_PRM_MIN: sel["min"]["answer"],
        M_PRM_PROD: sel["product"]["answer"],
        M_PRM_PEN: sel["step_penalty@2.0"]["answer"],
    }
    if BONUS_KEY in sel:
        preds["PRM Best-of-5 (product_logprob) [bonus]"] = sel[BONUS_KEY]["answer"]
    return preds


def _acc(flags):
    return round(100.0 * sum(flags) / len(flags), 2) if flags else 0.0


def summarize(out_dir):
    path = os.path.join(out_dir, "per_problem.jsonl")
    records = sorted(read_jsonl(path), key=lambda r: r["index"])
    if not records:
        print(f"No records in {path}")
        return
    n = len(records)
    gt = [r["ground_truth_answer"] for r in records]

    def correct(preds):
        return [p is not None and p == g for p, g in zip(preds, gt)]

    preds = [predictions_for(r) for r in records]
    sels = [selections_for(r) for r in records]

    greedy_tok = sum(r["greedy"]["num_tokens"] for r in records)
    sc5_tok = sum(sum(c["num_tokens"] for c in r["candidates"][:N_SHARED]) for r in records)
    sc15_tok = sum(sum(c["num_tokens"] for c in r["candidates"][:N_MATCHED]) for r in records)
    judge_calls = sum(r["prm"]["judge_calls"] for r in records)
    judge_tok = sum(r["prm"]["judge_tokens"] for r in records)

    t_greedy = sum(r["timing"]["greedy_s"] for r in records)
    t_sc5 = sum(r["timing"]["sc5_sampling_s"] for r in records)
    t_sc15x = sum(r["timing"]["sc15_extra_sampling_s"] for r in records)
    t_judge = sum(r["timing"]["judging_s"] for r in records)

    def t_sel(key):
        return sum(r["timing"]["select_s"].get(key, 0.0) for r in records)

    def row(method, gen_calls, j_calls, gen_tok, j_tok, t):
        return {
            "Method": method,
            "Accuracy (%)": _acc(correct([p[method] for p in preds])),
            "Generation Calls": gen_calls,
            "Judge Calls": j_calls,
            "Total LLM Calls": gen_calls + j_calls,
            "Avg Gen Tokens / Problem": round(gen_tok / n, 2),
            "Avg Judge Tokens / Problem": round(j_tok / n, 2),
            "Total Time (s)": round(t, 2),
        }

    summary = [
        row(M_GREEDY, n, 0, greedy_tok, 0, t_greedy),
        row(M_SC5, N_SHARED * n, 0, sc5_tok, 0, t_sc5),
        row(M_SC15, N_MATCHED * n, 0, sc15_tok, 0, t_sc5 + t_sc15x),
        row(M_PRM_MIN, N_SHARED * n, judge_calls, sc5_tok, judge_tok, t_sc5 + t_judge + t_sel("min")),
        row(M_PRM_PROD, N_SHARED * n, judge_calls, sc5_tok, judge_tok, t_sc5 + t_judge + t_sel("product")),
        row(M_PRM_PEN, N_SHARED * n, judge_calls, sc5_tok, judge_tok,
            t_sc5 + t_judge + t_sel("step_penalty@2.0")),
    ]
    save_results_summary(summary, os.path.join(out_dir, "results_summary.json"))

    ablation = []
    for key, strat, lam in SELECTIONS:
        ablation.append({
            "Strategy": strat,
            "Lambda": lam if strat == "step_penalty" else None,
            "Accuracy (%)": _acc(correct([s[key]["answer"] for s in sels])),
            "Tie Rate (%)": _acc([s[key]["num_tied"] > 1 for s in sels]),
        })
    save_aggregation_ablation(ablation, os.path.join(out_dir, "aggregation_ablation.json"))

    curve = []
    for N in range(1, N_MATCHED + 1):
        votes = [compute_majority_vote([c["answer"] for c in r["candidates"][:N]]) for r in records]
        toks = sum(sum(c["num_tokens"] for c in r["candidates"][:N]) for r in records)
        curve.append({"N": N, "Accuracy (%)": _acc(correct(votes)),
                      "Avg Gen Tokens / Problem": round(toks / n, 2)})
    save_sc_scaling_curve(curve, os.path.join(out_dir, "sc_scaling_curve.json"))

    # Sanity: the curve must agree with the SC rows by construction.
    assert curve[N_SHARED - 1]["Accuracy (%)"] == summary[1]["Accuracy (%)"]
    assert curve[N_MATCHED - 1]["Accuracy (%)"] == summary[2]["Accuracy (%)"]

    if all(BONUS_KEY in s for s in sels):
        t_bonus = sum((r["timing"].get("bonus_logprob_s") or 0.0) for r in records)
        bonus = {
            "Strategy": BONUS_KEY,
            "Method": "full-sequence log-prob of ' VALID' vs ' INVALID' after 'Verdict:'",
            "Accuracy (%)": _acc(correct([s[BONUS_KEY]["answer"] for s in sels])),
            "Tie Rate (%)": _acc([s[BONUS_KEY]["num_tied"] > 1 for s in sels]),
            "Extra Time (s)": round(t_bonus, 2),
        }
        with open(os.path.join(out_dir, "bonus_logprob.json"), "w") as f:
            json.dump(bonus, f, indent=2)

    print(f"\n=== Results over {n} problems ===")
    for s in summary:
        print(f"  {s['Method']:<42} acc={s['Accuracy (%)']:6.2f}%  calls={s['Total LLM Calls']:<6} "
              f"gen_tok/prob={s['Avg Gen Tokens / Problem']:<8} judge_tok/prob={s['Avg Judge Tokens / Problem']:<8} "
              f"time={s['Total Time (s)']}s")
    print("  Ablation:", json.dumps(ablation))
    print("  SC curve:", [c["Accuracy (%)"] for c in curve])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/gsm8k_test_50.jsonl")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge-batch-size", type=int, default=8)
    ap.add_argument("--no-bonus", action="store_true", help="skip the optional logprob bonus")
    ap.add_argument("--fresh", action="store_true", help="delete per_problem.jsonl and start over")
    ap.add_argument("--check-format", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--mock", action="store_true", help="offline dry run with fake models")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    per_problem = os.path.join(args.out_dir, "per_problem.jsonl")

    if args.summarize_only:
        summarize(args.out_dir)
        return

    problems = read_jsonl(args.data)
    if args.limit:
        problems = problems[: args.limit]

    if args.mock:
        random.seed(5326)
        gen_fn, judge_one, judge_batch, prob_fn = make_mock_fns(problems)
    else:
        from src.utils import (load_models, make_hf_batched_judge_fn, make_hf_generate_fn,
                               make_hf_judge_fn, make_verdict_prob_fn)
        t = time.perf_counter()
        gen_model, gen_tok, judge_model, judge_tok = load_models("cuda:0", "cuda:1")
        print(f"Models loaded in {time.perf_counter() - t:.1f}s")
        gen_fn = make_hf_generate_fn(gen_model, gen_tok)
        judge_one = make_hf_judge_fn(judge_model, judge_tok)
        judge_batch = make_hf_batched_judge_fn(judge_model, judge_tok, batch_size=args.judge_batch_size)
        prob_fn = make_verdict_prob_fn(judge_model, judge_tok, batch_size=args.judge_batch_size)
    if args.no_bonus:
        prob_fn = None

    if args.check_format:
        ok, none_ids = 0, []
        for idx, p in enumerate(problems):
            g = run_greedy(question=p["question"], generate_fn=gen_fn)
            has_steps = len(g["steps"]) > 0
            if g["answer"] is not None and has_steps:
                ok += 1
            else:
                none_ids.append(p["id"])
                print(f"--- non-compliant {p['id']} ---\n{g['completion'][-400:]}\n")
        print(f"Greedy delimiter compliance: {ok}/{len(problems)} ({100 * ok / len(problems):.1f}%)")
        print("Non-compliant:", none_ids)
        return

    if args.fresh and os.path.exists(per_problem):
        os.remove(per_problem)
    done = {r["id"] for r in read_jsonl(per_problem)}
    if done:
        print(f"Resuming: {len(done)} problems already in {per_problem}")

    run_start = time.perf_counter()
    for idx, p in enumerate(problems):
        if p["id"] in done:
            continue
        t = time.perf_counter()
        rec = run_problem(idx, p, gen_fn, judge_one, judge_batch, prob_fn)
        write_jsonl_line(per_problem, rec)
        c = rec["correct"]
        print(f"[{idx + 1:02d}/{len(problems)}] {p['id']} gt={p['ground_truth_answer']} "
              f"greedy={rec['predictions'][M_GREEDY]}({'+' if c[M_GREEDY] else '-'}) "
              f"sc5={rec['predictions'][M_SC5]}({'+' if c[M_SC5] else '-'}) "
              f"sc15={rec['predictions'][M_SC15]}({'+' if c[M_SC15] else '-'}) "
              f"prm_min={rec['predictions'][M_PRM_MIN]}({'+' if c[M_PRM_MIN] else '-'}) "
              f"judge_calls={rec['prm']['judge_calls']} [{time.perf_counter() - t:.1f}s]", flush=True)
    print(f"Run time this session: {time.perf_counter() - run_start:.1f}s")

    summarize(args.out_dir)


if __name__ == "__main__":
    main()
