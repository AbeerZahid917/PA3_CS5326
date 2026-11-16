"""
analyze.py - Report helper (CPU only, run AFTER run_all.py). Produces the numbers and
figures the report asks for; the written analysis itself is yours to write.

  python analyze.py                       # stats + plots + failure-case candidates
  python analyze.py --make-audit          # export >=30 judged steps (>=10 INVALID) to label
  python analyze.py --score-audit         # confusion matrix + precision/recall from your labels

Outputs (in outputs/analysis/):
  report_stats.json        bootstrap 95% CIs, paired win/loss, format failures, 10x-greedy budget
  scaling_calls.png        Accuracy vs Total LLM Calls / problem (SC N=1..15 + PRM points)
  scaling_tokens.png       Accuracy vs Total Output Tokens / problem (gen + judge)
  failure_candidates.json  problems that likely show judge FP / judge FN / mode collapse
  judge_audit.csv          steps to hand-label (fill the human_label column with VALID/INVALID)
"""
import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_all import (M_GREEDY, M_PRM_MIN, M_PRM_PEN, M_PRM_PROD, M_SC5, M_SC15, N_SHARED,
                     predictions_for)
from src.generation import compute_majority_vote
from src.utils import read_jsonl

SEED = 5326
METHODS = [M_GREEDY, M_SC5, M_SC15, M_PRM_MIN, M_PRM_PROD, M_PRM_PEN]


def load(out_dir):
    recs = sorted(read_jsonl(os.path.join(out_dir, "per_problem.jsonl")), key=lambda r: r["index"])
    if not recs:
        sys.exit(f"No per_problem.jsonl in {out_dir}; run run_all.py first.")
    summary = json.load(open(os.path.join(out_dir, "results_summary.json")))
    curve = json.load(open(os.path.join(out_dir, "sc_scaling_curve.json")))
    return recs, summary, curve


def bootstrap_ci(flags, n_boot=1000, seed=SEED):
    rng = random.Random(seed)
    n = len(flags)
    accs = []
    for _ in range(n_boot):
        s = [flags[rng.randrange(n)] for _ in range(n)]
        accs.append(100.0 * sum(s) / n)
    accs.sort()
    return round(accs[int(0.025 * n_boot)], 2), round(accs[int(0.975 * n_boot) - 1], 2)


def paired(a_flags, b_flags):
    a_only = sum(1 for a, b in zip(a_flags, b_flags) if a and not b)
    b_only = sum(1 for a, b in zip(a_flags, b_flags) if b and not a)
    both = sum(1 for a, b in zip(a_flags, b_flags) if a and b)
    neither = sum(1 for a, b in zip(a_flags, b_flags) if not a and not b)
    return {"A wins": a_only, "B wins": b_only, "both correct": both, "both wrong": neither}


def stats(out_dir):
    recs, summary, curve = load(out_dir)
    n = len(recs)
    preds = [predictions_for(r) for r in recs]
    gt = [r["ground_truth_answer"] for r in recs]
    flags = {m: [p[m] is not None and p[m] == g for p, g in zip(preds, gt)] for m in METHODS}

    out = {"n_problems": n, "accuracy_ci95": {}, "paired": {}, "format_failures": {}}
    for m in METHODS:
        acc = round(100.0 * sum(flags[m]) / n, 2)
        lo, hi = bootstrap_ci(flags[m])
        out["accuracy_ci95"][m] = {"accuracy": acc, "ci_low": lo, "ci_high": hi}

    for prm in (M_PRM_MIN, M_PRM_PROD, M_PRM_PEN):
        out["paired"][f"{prm} (A) vs {M_SC5} (B)"] = paired(flags[prm], flags[M_SC5])
        out["paired"][f"{prm} (A) vs {M_SC15} (B)"] = paired(flags[prm], flags[M_SC15])

    # Format compliance: answers that came back None
    out["format_failures"] = {
        "greedy_none": sum(r["greedy"]["answer"] is None for r in recs),
        "sampled_none": sum(c["answer"] is None for r in recs for c in r["candidates"]),
        "sampled_total": sum(len(r["candidates"]) for r in recs),
        "problems_where_SC5_all_none": sum(
            all(c["answer"] is None for c in r["candidates"][:N_SHARED]) for r in recs),
    }

    # Mean steps per path (K-bar) and verdict distribution
    shared = [c for r in recs for c in r["candidates"][:N_SHARED]]
    steps_total = sum(len(c["steps"]) for c in shared)
    invalid = sum(1 for c in shared for s in c.get("step_scores", []) if s == -1.0)
    out["judge"] = {
        "K_bar_mean_steps_per_path": round(steps_total / max(1, len(shared)), 2),
        "steps_judged": steps_total,
        "steps_marked_INVALID": invalid,
        "pct_INVALID": round(100.0 * invalid / max(1, steps_total), 2),
        "paths_all_VALID": sum(1 for c in shared if c.get("step_scores") and min(c["step_scores"]) == 1.0),
        "paths_total": len(shared),
    }

    # Section 4: 10x greedy budget
    by = {s["Method"]: s for s in summary}
    g_tok = by[M_GREEDY]["Avg Gen Tokens / Problem"]
    budget = 10 * g_tok
    prm_spend = by[M_PRM_MIN]["Avg Gen Tokens / Problem"] + by[M_PRM_MIN]["Avg Judge Tokens / Problem"]
    within = [c for c in curve if c["Avg Gen Tokens / Problem"] <= budget]
    out["budget_10x_greedy"] = {
        "greedy_gen_tokens_per_problem": g_tok,
        "budget_tokens_per_problem": round(budget, 2),
        "prm_bo5_tokens_per_problem(gen+judge)": round(prm_spend, 2),
        "prm_fits_budget": prm_spend <= budget,
        "largest_SC_N_within_budget": within[-1]["N"] if within else None,
        "SC_accuracy_at_that_N": within[-1]["Accuracy (%)"] if within else None,
        "judge_to_gen_call_ratio": round(by[M_PRM_MIN]["Judge Calls"] / max(1, by[M_PRM_MIN]["Generation Calls"]), 2),
        "avg_judge_tokens_per_call": round(by[M_PRM_MIN]["Avg Judge Tokens / Problem"] * n
                                           / max(1, by[M_PRM_MIN]["Judge Calls"]), 2),
        "avg_gen_tokens_per_call": round(by[M_SC5]["Avg Gen Tokens / Problem"] / N_SHARED, 2),
        "note": "Judge OUTPUT tokens are few, but every judge call re-reads a long prompt "
                "(problem + all preceding steps) through a 7B model: prefill cost is not in these counts.",
    }

    adir = os.path.join(out_dir, "analysis")
    os.makedirs(adir, exist_ok=True)
    with open(os.path.join(adir, "report_stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    plots(adir, summary, curve, n)
    failure_candidates(adir, recs)


def plots(adir, summary, curve, n):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: skipping plots")
        return
    by = {s["Method"]: s for s in summary}
    prm = [(M_PRM_MIN, "o", "PRM BoN-5 (min)"), (M_PRM_PROD, "s", "PRM BoN-5 (product)"),
           (M_PRM_PEN, "^", "PRM BoN-5 (step_penalty, λ=2)")]

    # 1. Accuracy vs total LLM calls per problem
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot([c["N"] for c in curve], [c["Accuracy (%)"] for c in curve], "-o", ms=3, label="Self-consistency N=1..15")
    for m, mk, lab in prm:
        ax.scatter(by[m]["Total LLM Calls"] / n, by[m]["Accuracy (%)"], marker=mk, s=60, label=lab, zorder=3)
    ax.scatter(1, by[M_GREEDY]["Accuracy (%)"], marker="x", s=60, color="k", label="Greedy", zorder=3)
    ax.set_xlabel("Total LLM calls per problem (generation + judge)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs LLM calls (GSM8K-50)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(adir, "scaling_calls.png"), dpi=200)
    plt.close(fig)

    # 2. Accuracy vs total output tokens per problem
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot([c["Avg Gen Tokens / Problem"] for c in curve], [c["Accuracy (%)"] for c in curve], "-o", ms=3,
            label="Self-consistency N=1..15")
    for m, mk, lab in prm:
        ax.scatter(by[m]["Avg Gen Tokens / Problem"] + by[m]["Avg Judge Tokens / Problem"],
                   by[m]["Accuracy (%)"], marker=mk, s=60, label=lab, zorder=3)
    ax.scatter(by[M_GREEDY]["Avg Gen Tokens / Problem"], by[M_GREEDY]["Accuracy (%)"], marker="x", s=60,
               color="k", label="Greedy", zorder=3)
    ax.axvline(10 * by[M_GREEDY]["Avg Gen Tokens / Problem"], ls="--", color="gray", lw=1, label="10× greedy budget")
    ax.set_xlabel("Total output tokens per problem (generated + judge)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs output tokens (GSM8K-50)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(adir, "scaling_tokens.png"), dpi=200)
    plt.close(fig)
    print(f"Plots written to {adir}/")


def failure_candidates(adir, recs):
    """Heuristic pointers. YOU must read the traces and confirm each case by hand."""
    fp, fn, collapse = [], [], []
    for r in recs:
        g = r["ground_truth_answer"]
        for i, c in enumerate(r["candidates"][:N_SHARED]):
            sc = c.get("step_scores") or []
            if not sc:
                continue
            if c["answer"] is not None and c["answer"] != g and min(sc) == 1.0:
                fp.append({"id": r["id"], "cand": i, "answer": c["answer"], "gt": g,
                           "why": "wrong final answer but every step judged VALID -> some flawed step was passed"})
            if c["answer"] == g and min(sc) == -1.0:
                bad = [t for t, s in enumerate(sc) if s == -1.0]
                fn.append({"id": r["id"], "cand": i, "invalid_steps": bad,
                           "why": "correct answer but a step judged INVALID -> check if the step is actually fine"})
        answers = [c["answer"] for c in r["candidates"]]
        vote = compute_majority_vote(answers)
        if vote is not None and vote != g:
            share = sum(a == vote for a in answers) / len(answers)
            if share >= 0.6:
                collapse.append({"id": r["id"], "wrong_majority": vote, "gt": g, "share_of_15": round(share, 2)})
    out = {"judge_false_positive_candidates": fp, "judge_false_negative_candidates": fn,
           "mode_collapse_candidates": sorted(collapse, key=lambda x: -x["share_of_15"])}
    with open(os.path.join(adir, "failure_candidates.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Failure candidates: FP={len(fp)} FN={len(fn)} collapse={len(collapse)} "
          f"(see {adir}/failure_candidates.json)")


def make_audit(out_dir, n_total=40, n_invalid=15):
    recs, _, _ = load(out_dir)
    rng = random.Random(SEED)
    rows_valid, rows_invalid = [], []
    for r in recs:
        for i, c in enumerate(r["candidates"][:N_SHARED]):
            for t, s in enumerate(c.get("step_scores") or []):
                row = {
                    "id": r["id"], "cand": i, "step_idx": t + 1, "gt": r["ground_truth_answer"],
                    "question": r["question"],
                    "preceding_steps": " | ".join(f"Step {k + 1}: {x}" for k, x in enumerate(c["steps"][:t])),
                    "current_step": c["steps"][t],
                    "judge_verdict": "VALID" if s == 1.0 else "INVALID",
                    "judge_rationale": c["rationales"][t],
                    "human_label": "",
                }
                (rows_valid if s == 1.0 else rows_invalid).append(row)
    rng.shuffle(rows_valid)
    rng.shuffle(rows_invalid)
    k_inv = min(n_invalid, len(rows_invalid))
    rows = rows_invalid[:k_inv] + rows_valid[: max(0, n_total - k_inv)]
    rng.shuffle(rows)
    path = os.path.join(out_dir, "analysis", "judge_audit.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} steps ({k_inv} judged INVALID) to {path}. "
          f"Fill human_label with VALID or INVALID (judge each step under local validity), "
          f"then run: python analyze.py --score-audit")


def score_audit(out_dir):
    path = os.path.join(out_dir, "analysis", "judge_audit.csv")
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["human_label"].strip()]
    # Positive class = INVALID (we are measuring the judge as an error detector).
    tp = sum(r["judge_verdict"] == "INVALID" and r["human_label"].strip().upper() == "INVALID" for r in rows)
    fp = sum(r["judge_verdict"] == "INVALID" and r["human_label"].strip().upper() == "VALID" for r in rows)
    fn = sum(r["judge_verdict"] == "VALID" and r["human_label"].strip().upper() == "INVALID" for r in rows)
    tn = sum(r["judge_verdict"] == "VALID" and r["human_label"].strip().upper() == "VALID" for r in rows)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    res = {"labeled": len(rows), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
           "precision_INVALID": round(prec, 3), "recall_INVALID": round(rec, 3),
           "note": "positive class = INVALID. FP here = sound step the judge rejected; "
                   "FN = flawed step the judge passed (the report's 'judge false positive')."}
    with open(os.path.join(out_dir, "analysis", "judge_audit_scores.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--make-audit", action="store_true")
    ap.add_argument("--score-audit", action="store_true")
    a = ap.parse_args()
    if a.make_audit:
        make_audit(a.out_dir)
    elif a.score_audit:
        score_audit(a.out_dir)
    else:
        stats(a.out_dir)
