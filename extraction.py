"""
src/extraction.py - Step and answer extraction utilities.

Design notes
------------
* parse_steps keeps the markdown-tolerant regex given in the handout, plus one
  extra clean-up: if the model writes "Step 4: ... = 18. Final Answer: 18" on a
  single line, the trailing "Final Answer ..." is cut off the step text so the
  judge never sees the answer delimiter as part of a reasoning step.
* extract_answer is deliberately strict: it only reads the number that follows
  the LAST "Final Answer" delimiter. There is no fallback to "the last integer
  in the text" -- format failures must surface as None (report Section 3).
"""
import re

# Markdown-tolerant step pattern from the assignment handout (Listing 1).
_STEP_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)(.*?)"
    r"(?=(?:\n\s*[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)|(?:\n\s*[#*`\s]*Final Answer)|$)",
    flags=re.DOTALL | re.IGNORECASE,
)

# "Final Answer" delimiter: case-insensitive, any whitespace between the words.
_FINAL_ANSWER_RE = re.compile(r"final\s*answer", flags=re.IGNORECASE)

# Same delimiter used inside a step line (optionally bolded), to trim it off.
# (colon required, so a step like "so the final answer is 8 + 2" is left intact).
_INLINE_FINAL_RE = re.compile(r"[*#`\s]*final\s*answer[*\s]*:.*$", flags=re.IGNORECASE | re.DOTALL)

# A number right at the start of the text that follows the delimiter:
#   optional minus, optional "$", digits with optional thousands commas,
#   optional decimal part.
_NUMBER_RE = re.compile(r"^(-)?\s*\$?\s*(\d[\d,]*)(\.\d+)?")


def parse_steps(completion_text: str) -> list[str]:
    """
    Parses completion text and returns a list of individual step strings.
    Must tolerate markdown decoration around labels (e.g. '**Step 1:**', '### Step 2:')
    and strip leftover '*' or '#' from captured text.
    """
    if not completion_text or not isinstance(completion_text, str):
        return []

    try:
        matches = _STEP_PATTERN.findall(completion_text)
    except Exception:  # pragma: no cover - regex on str should not fail
        return []

    cleaned = []
    for m in matches:
        text = m.strip()
        # Drop an inline "Final Answer: ..." that shares the line with the last step.
        text = _INLINE_FINAL_RE.sub("", text)
        text = text.strip().strip("*#`").strip()
        if text:
            cleaned.append(text)
    return cleaned


def extract_answer(completion_text: str) -> int | None:
    """
    Extracts the final numeric integer from the completion.

    Required Behaviors:
      - 'Final Answer: 18' -> 18
      - 'Final Answer: $1,234' -> 1234
      - 'Final Answer: 18.00' -> 18
      - 'Final Answer: 18.5' -> None (non-integer)
      - 'Final Answer: -5' -> -5
      - 'Final Answer: 18 dollars.' -> 18
      - '**Final Answer:** 18' -> 18
      - 'final answer 18' (case-insensitive, colon optional) -> 18
      - Multiple 'Final Answer' lines -> use the LAST one
      - No 'Final Answer' present -> None
      - Empty string or None-like garbage -> None, and never raise
    """
    try:
        if not completion_text or not isinstance(completion_text, str):
            return None

        delimiters = list(_FINAL_ANSWER_RE.finditer(completion_text))
        if not delimiters:
            return None

        # Text after the LAST delimiter (discards earlier scratchpad attempts).
        tail = completion_text[delimiters[-1].end():]

        # Strip decoration between the delimiter and the value:
        #   colons, markdown bold/headers/backticks, whitespace, "is", "=".
        tail = re.sub(r"^[\s:*#`_=]*", "", tail)
        tail = re.sub(r"^(?:is|=)\s*", "", tail, flags=re.IGNORECASE)
        # Tolerate a LaTeX box around the value: \boxed{18} or $\boxed{18}$.
        tail = re.sub(r"^\$?\s*\\boxed\{\s*", "", tail)

        m = _NUMBER_RE.match(tail)
        if not m:
            return None

        sign, int_part, frac_part = m.group(1), m.group(2), m.group(3)
        digits = int_part.replace(",", "")
        if not digits.isdigit():
            return None

        if frac_part:
            # Accept integer-valued decimals (18.00) but reject 18.5.
            if set(frac_part[1:]) != {"0"}:
                return None

        value = int(digits)
        return -value if sign else value
    except Exception:
        # Contract: never raise on malformed input.
        return None
