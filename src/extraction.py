"""
src/extraction.py - Step and answer extraction utilities.
"""
import re


def parse_steps(completion_text: str) -> list[str]:
    """
    Parses completion text and returns a list of individual step strings.
    Must tolerate markdown decoration around labels (e.g. '**Step 1:**', '### Step 2:')
    and strip leftover '*' or '#' from captured text.
    """
    if not completion_text or not isinstance(completion_text, str):
        return []

    # Pattern matching Step markers with optional markdown decoration (#, *, `, :, .)
    pattern = r"(?:^|\n)\s*(?:[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)(.*?)(?=(?:\n\s*[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)|(?:\n\s*[#*`\s]*Final Answer)|$)"
    matches = re.findall(pattern, completion_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = []
    for m in matches:
        text = m.strip().strip("*#`").strip()
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

        # 1. Find every "Final Answer" delimiter (case-insensitive, colon optional).
        delimiters = list(re.finditer(r"final\s*answer", completion_text, flags=re.IGNORECASE))
        if not delimiters:
            return None  # no delimiter -> None (no fallback to random integers)

        # 2. Keep only the text after the LAST delimiter (earlier ones were scratch work).
        tail = completion_text[delimiters[-1].end():]

        # 3. Strip decoration between the delimiter and the number:
        #    colons, markdown (** # ` _), '=', whitespace, and a leading "is".
        tail = re.sub(r"^[\s:*#`_=]*", "", tail)
        tail = re.sub(r"^(?:is|=)\s*", "", tail, flags=re.IGNORECASE)

        # 4. Read the number at the start: optional minus, optional $, digits with
        #    optional thousands commas, optional decimal part. Trailing units are ignored.
        m = re.match(r"^(-)?\s*\$?\s*(\d[\d,]*)(\.\d+)?", tail)
        if not m:
            return None
        sign, int_part, frac_part = m.group(1), m.group(2), m.group(3)

        # 5. Decimals are only accepted if they are whole numbers (18.00 ok, 18.5 not).
        if frac_part and set(frac_part[1:]) != {"0"}:
            return None

        value = int(int_part.replace(",", ""))
        return -value if sign else value
    except Exception:
        return None  # contract: never raise