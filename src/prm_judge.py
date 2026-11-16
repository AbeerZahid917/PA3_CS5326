"""
src/prm_judge.py - PRM judge prompt construction, parsing, and step scoring.

The 7B judge acts as a stand-in Process Reward Model: for every step s_t of a
candidate it answers VALID / INVALID under *local validity* semantics, i.e.
assuming steps s_1..s_{t-1} are correct as written. The binary verdict is mapped
to r_t in {+1.0, -1.0}.
"""
import re
from typing import Any, Callable

JudgeFn = Callable[[str], dict]
# judge_fn(judge_prompt) -> {"text": str, "num_tokens": int}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
# Output order: the judge writes the one-sentence Rationale FIRST and the Verdict
# SECOND. With greedy decoding this lets the judge re-check the arithmetic in
# words before committing to a label (a tiny chain-of-thought), which in practice
# reduces careless VALID stamps. parse_judge_decision does not depend on order.
_JUDGE_TEMPLATE = """You are a strict and careful math teacher. You are checking ONE step of a student's step-by-step solution to a grade-school math word problem.

Problem:
{question}

Preceding steps:
{preceding}

Current Step: {current}

How to judge:
- Assume all preceding steps are correct, even if they contain mistakes. Judge ONLY the Current Step.
- The Current Step is VALID if every calculation in it is arithmetically correct AND it follows logically from the problem statement together with the preceding steps as written (right quantities, right operation, facts read correctly from the problem).
- The Current Step is INVALID if it contains any arithmetic error, uses a wrong number or a wrong operation, misreads or invents information, or does not follow from the problem and the preceding steps.
- A step that simply restates information from the problem or from a preceding step is VALID if it restates it accurately.
- Recompute every number in the Current Step yourself before deciding.

Answer with exactly two lines and nothing else:
Rationale: <one short sentence that checks the step's arithmetic and logic>
Verdict: <VALID or INVALID>

The second line must be exactly "Verdict: VALID" if the step is correct, or exactly "Verdict: INVALID" if it is not."""


def build_prm_judge_prompt(question: str, preceding_steps: list[str], current_step: str) -> str:
    """
    Builds the verification prompt for the PRM judge to classify current_step
    as VALID or INVALID given the problem question and preceding_steps.

    Judge Semantics (Local Validity):
    The judge evaluates current_step ASSUMING ALL PRECEDING STEPS ARE CORRECT.
    A step is VALID if its arithmetic is sound and it follows logically from the
    problem statement plus preceding steps as written. An error in an earlier step
    does NOT make subsequent steps invalid by itself.

    Autograder Literal Checks:
    - Preceding steps are rendered as 'Step 1: <text>', 'Step 2: <text>', ... in order.
    - The current step is rendered with the label 'Current Step:'.
    - The prompt contains the phrase 'assume all preceding steps are correct' (case-insensitive).
    - The prompt contains the literal strings 'Verdict: VALID', 'Verdict: INVALID', and 'Rationale:'.
    - With an empty preceding_steps list, the prompt states '(no preceding steps)'.
    """
    steps = list(preceding_steps or [])
    if steps:
        preceding = "\n".join(f"Step {i + 1}: {str(s).strip()}" for i, s in enumerate(steps))
    else:
        preceding = "(no preceding steps)"
    return _JUDGE_TEMPLATE.format(
        question=(question or "").strip(),
        preceding=preceding,
        current=str(current_step or "").strip(),
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_VERDICT_LINE_RE = re.compile(r"verdict\s*:\s*\**\s*(VALID|INVALID)\b", re.IGNORECASE)
_NEGATIVE_RE = re.compile(r"\bINVALID\b|\bNOT\s+VALID\b", re.IGNORECASE)
_POSITIVE_RE = re.compile(r"\bVALID\b", re.IGNORECASE)
_RATIONALE_RE = re.compile(r"rationale\**\s*:\s*\**", re.IGNORECASE)
# Sentence end: . ! or ? followed by whitespace or end of string ("2.5" is not split).
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")


def parse_judge_decision(judge_output: str) -> tuple[float, str]:
    r"""
    Parses the judge model's raw text completion into (score, rationale).

    Rules:
      1. If a line matches /verdict\s*:\s*\**\s*(VALID|INVALID)\b/i, use it.
         If several exist, use the first.
      2. Otherwise, if standalone word INVALID or phrase 'NOT VALID' appears
         (case-insensitive, word boundaries) -> -1.0.
      3. Otherwise, if standalone word VALID appears -> +1.0.
      4. Otherwise -> -1.0 (unparseable defaults to INVALID).

    Warning: A naive '"VALID" in text' check is wrong because "INVALID" contains "VALID".

    Rationale: Text after 'Rationale:' up to the first sentence end (".", "!", or "?"
    followed by whitespace or end of string, so decimals like "2.5" are not split),
    or a newline, whichever comes first, stripped. If absent, returns empty string "".
    """
    if not judge_output or not isinstance(judge_output, str):
        return -1.0, ""

    # ---- score ----
    m = _VERDICT_LINE_RE.search(judge_output)
    if m:
        score = 1.0 if m.group(1).upper() == "VALID" else -1.0
    elif _NEGATIVE_RE.search(judge_output):
        score = -1.0
    elif _POSITIVE_RE.search(judge_output):
        score = 1.0
    else:
        score = -1.0

    # ---- rationale ----
    rationale = ""
    r = _RATIONALE_RE.search(judge_output)
    if r:
        rest = judge_output[r.end():]
        rest = rest.split("\n", 1)[0]          # stop at newline
        e = _SENTENCE_END_RE.search(rest)      # stop at first sentence end
        if e:
            rest = rest[: e.end()]
        rationale = rest.strip()

    return score, rationale


# ---------------------------------------------------------------------------
# Path scoring
# ---------------------------------------------------------------------------
def _resolve_judge_fn(judge_model: Any, judge_tokenizer: Any, judge_fn: JudgeFn | None) -> JudgeFn:
    if judge_fn is not None:
        return judge_fn
    if judge_model is None or judge_tokenizer is None:
        raise ValueError("Provide either judge_fn or both judge_model and judge_tokenizer.")
    from .utils import make_hf_judge_fn  # lazy import
    return make_hf_judge_fn(judge_model, judge_tokenizer)


def score_path(
    judge_model: Any = None,
    judge_tokenizer: Any = None,
    question: str = "",
    candidate: dict[str, Any] | None = None,
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Sequentially scores each step in candidate["steps"] using judge_fn.
    Step t is judged with preceding_steps = steps[:t].

    The student's code MUST call build_prm_judge_prompt to build the string passed to judge_fn,
    and MUST call parse_judge_decision on the returned text.

    Returns:
      step_scores, rationales, judge_calls, judge_tokens
      (+ judge_outputs: raw judge texts, kept for the failure/audit analysis)
    """
    steps = list((candidate or {}).get("steps") or [])
    if not steps:
        return {"step_scores": [], "rationales": [], "judge_calls": 0,
                "judge_tokens": 0, "judge_outputs": []}

    judge = _resolve_judge_fn(judge_model, judge_tokenizer, judge_fn)

    step_scores: list[float] = []
    rationales: list[str] = []
    raw_outputs: list[str] = []
    judge_tokens = 0
    for t, step in enumerate(steps):
        prompt = build_prm_judge_prompt(question, steps[:t], step)
        out = judge(prompt) or {}
        text = out.get("text", "") or ""
        score, rationale = parse_judge_decision(text)
        step_scores.append(score)
        rationales.append(rationale)
        raw_outputs.append(text)
        judge_tokens += int(out.get("num_tokens", 0) or 0)

    return {
        "step_scores": step_scores,
        "rationales": rationales,
        "judge_calls": len(steps),
        "judge_tokens": judge_tokens,
        "judge_outputs": raw_outputs,
    }