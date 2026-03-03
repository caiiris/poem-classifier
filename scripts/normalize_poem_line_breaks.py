"""
Apply line-break normalization to the main dataset.

1. All poems: (a) where . , ; : ! ? or ellipsis (...) is immediately followed by a letter (no space),
   insert a newline; (b) where em/en-dash (— –) is immediately followed by a capitalized function word,
   insert a newline; (c) where a word is entirely uppercase (3+ letters) with a space before and after,
   break the line before and after it;    (d) where a Roman numeral II or above (II, III, IV, …) with optional period is
   followed by space and a capitalized word, put the numeral on its own line; I alone is excluded;
   (e) where "I." appears in the middle of a line (space before it), remove the period so it reads as first-person "I".

2. Pre-1800 poems only: (a) where two uppercase letters appear in one word but are
   not adjacent, split between them ("WordAnother" → "Word\nAnother");
   (b) where a capitalized conjunction, preposition, or archaic function word follows a non-capitalized word,
   break the line before it ("word And" → "word\nAnd");
   (c) where punctuation is followed by space then a capital letter, break after the punctuation ("food, And" → "food,\nAnd");
   (d) where 3+ consecutive lines are each "one word and a comma" (list-like), merge those lines into one;

3. Pre-1900 poems only: replace multiple consecutive spaces with a newline
   (split on multiple spaces).

Reads and overwrites: data/PoetryFoundationData_with_year.csv

Do NOT add that split for post-1800 poems (would inflate cap_rate); we do it for pre-1800 only (rule 2c).
"""

import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "PoetryFoundationData_with_year.csv"

# Period, comma, colon, semicolon, exclamation, question; if immediately followed by a letter (no space), insert newline.
# Do NOT split on "punctuation + space + capitalized word" (e.g. ". The" → ".\nThe"); that would inflate cap_rate.
PUNCT_NO_SPACE_RE = re.compile(r"([.,;:!?])([a-zA-Z])")
# Ellipsis (three dots or Unicode …) immediately followed by a letter → newline
ELLIPSIS_NO_SPACE_RE = re.compile(r"(\.\.\.|…)([a-zA-Z])")

# Two capitalized "words" concatenated (lowercase then uppercase) → split before the uppercase
UPPERCASE_SPLIT_RE = re.compile(r"([A-Za-z]*[a-z])([A-Z][A-Za-z]*)")

# Multiple spaces → newline (for pre-1900)
SPACES_TO_NEWLINE_RE = re.compile(r" {2,}")

# Pre-1800: capitalized conjunction/preposition after a lowercase letter → newline before it
CAP_CONJUNCTIONS = (
    "And", "But", "Or", "Nor", "For", "Yet", "So",
    "Then", "Therefore", "Thus", "Hence",
    "If", "When", "Where", "While", "Although", "Because",
    "Unless", "Until", "Whereas", "Whilst",
    "Also", "Nevertheless", "However", "Moreover", "Furthermore",
    "Either", "Neither", "Not",
)
CAP_PREPOSITIONS = (
    "In", "On", "At", "To", "For", "With", "By", "From", "Of",
    "Into", "Upon", "Within", "Without", "Between", "Among", "Amongst",
    "Above", "Below", "Before", "After", "Behind", "Beyond",
    "Through", "Throughout", "During", "Under", "Over",
    "Till", "Toward", "Towards", "About", "Against",
)
CAP_ARCHAIC = (
    "Nay", "Yea", "Verily", "Forsooth", "Ere", "Lest", "Albeit", "Howbeit",
    "Whence", "Whither", "Thither", "Thence", "Hither",
    "Betwixt", "Amidst", "Anon", "Oft", "Lo",
    "Henceforth", "Thenceforth", "Wherefore", "Prithee", "Perchance",
    "Forasmuch", "Inasmuch", "Notwithstanding", "Wheresoever",
)
CAP_FUNCTION_WORDS = tuple(dict.fromkeys(CAP_CONJUNCTIONS + CAP_PREPOSITIONS + CAP_ARCHAIC))  # dedupe e.g. For
CAP_FUNCTION_ALTERNATION = "|".join(re.escape(w) for w in CAP_FUNCTION_WORDS)
CAP_FUNCTION_AFTER_LOWER_RE = re.compile(
    rf"([a-z])\s+({CAP_FUNCTION_ALTERNATION})\b"
)
# Em/en-dash only when immediately followed by a capitalized function word (avoid splitting "word—something")
EM_EN_DASH_FUNCTION_RE = re.compile(
    rf"([—–])({CAP_FUNCTION_ALTERNATION})\b"
)
# Word entirely uppercase (3+ letters) with space before and after → put on its own line
ALL_CAPS_WORD_RE = re.compile(r"\s+([A-Z]{3,})\s+")
# Pre-1800 only: punctuation, space(s), capital letter → break after punctuation ("food, And" → "food,\nAnd")
PUNCT_SPACE_CAP_RE = re.compile(r"([.,;:!?])\s+([A-Z])")
# Pre-1800: line is exactly one word and a comma (list-like)
ONE_WORD_COMMA_RE = re.compile(r"^\s*\w+\s*,?\s*$")
# Roman numeral II or above (exclude I to avoid first-person "I") + optional period + space + capitalized word → put numeral on its own line
ROMAN_NUMERAL_RUN = r"(?:II|III|IV|V|VI{0,3}|VII|VIII|IX|X|XI{0,3}|XV|XVI{0,3}|XX)"
ROMAN_NUMERAL_SPACE_CAP_RE = re.compile(
    rf"(\s)({ROMAN_NUMERAL_RUN})(\.?)(\s+)([A-Z]\w*)",
    re.IGNORECASE,
)
# "I." in the middle of a line (preceded by space) → "I" (first-person pronoun)
I_PERIOD_MIDLINE_RE = re.compile(r"(\s)I\.(\s)")


def punctuation_no_space_to_newline(text: str) -> str:
    """Insert newline after . , ; : ! ? or ellipsis when next char is letter; after —/– only when next word is cap function word."""
    if not isinstance(text, str) or not text.strip():
        return text
    text = PUNCT_NO_SPACE_RE.sub(r"\1\n\2", text)
    text = ELLIPSIS_NO_SPACE_RE.sub(r"\1\n\2", text)
    text = EM_EN_DASH_FUNCTION_RE.sub(r"\1\n\2", text)
    return text


def all_caps_word_own_line(text: str) -> str:
    """Put any word that is entirely uppercase (3+ letters) with space before and after on its own line."""
    if not isinstance(text, str) or not text.strip():
        return text
    return ALL_CAPS_WORD_RE.sub(r"\n\1\n", text)


def split_uppercase_words_in_line(line: str) -> str:
    """Within a line, split between two non-adjacent uppercase letters (e.g. WordAnother → Word\\nAnother)."""
    if not line.strip():
        return line
    out = line
    while True:
        new_out = UPPERCASE_SPLIT_RE.sub(r"\1\n\2", out)
        if new_out == out:
            break
        out = new_out
    return out


def pre1800_uppercase_splits(text: str) -> str:
    """Apply uppercase-word splitting to each line of the poem."""
    if not isinstance(text, str) or not text.strip():
        return text
    lines = text.split("\n")
    return "\n".join(split_uppercase_words_in_line(line) for line in lines)


def pre1800_break_before_cap_conjunction(text: str) -> str:
    """Break line before capitalized conjunction/preposition when previous word is not capitalized."""
    if not isinstance(text, str) or not text.strip():
        return text
    return CAP_FUNCTION_AFTER_LOWER_RE.sub(r"\1\n\2", text)


def pre1800_punct_space_cap_to_newline(text: str) -> str:
    """Break line after punctuation when followed by space then capital letter (e.g. 'food, And' → 'food,\\nAnd')."""
    if not isinstance(text, str) or not text.strip():
        return text
    return PUNCT_SPACE_CAP_RE.sub(r"\1\n\2", text)


def _is_one_word_comma_line(line: str) -> bool:
    """True if line is just one word and optionally a comma (e.g. 'One,' or 'Two,')."""
    return bool(ONE_WORD_COMMA_RE.match(line))


# Undo: merge standalone "I." or "I" (one line) back so it's not on its own line (revert when we only do II+)
STANDALONE_I_RE = re.compile(r"\nI\.?\s*\n", re.IGNORECASE)


def roman_numeral_own_line(text: str) -> str:
    """Put Roman numeral II or above (II, III, IV, …) with optional period on its own line when followed by space and capital."""
    if not isinstance(text, str) or not text.strip():
        return text
    # First merge back any standalone I. or I that was split previously (undo I-only splits)
    text = STANDALONE_I_RE.sub(" I. ", text)
    # Then split only for II and above: " space II. space The" → "\nII.\nThe"
    return ROMAN_NUMERAL_SPACE_CAP_RE.sub(r"\n\2\3\n\5", text)


def pre1800_merge_list_like_lines(text: str) -> str:
    """Merge 3+ consecutive 'one word,' lines into a single line; leave rest of poem unchanged."""
    if not isinstance(text, str) or not text.strip():
        return text
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        # Count consecutive list-like lines (one word + comma)
        j = i
        while j < len(lines) and _is_one_word_comma_line(lines[j]):
            j += 1
        if j - i >= 3:
            # Merge: "One," "Two," "Three," → "One, Two, Three,"
            merged = " ".join(l.strip() for l in lines[i:j])
            out.append(merged)
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def spaces_to_newlines(text: str) -> str:
    """Replace 2+ consecutive spaces with a newline."""
    if not isinstance(text, str) or not text.strip():
        return text
    return SPACES_TO_NEWLINE_RE.sub("\n", text)


def main():
    df = pd.read_csv(CSV_PATH)
    poems = df["Poem"].fillna("")
    df["Poem"] = poems.apply(punctuation_no_space_to_newline)
    df["Poem"] = df["Poem"].apply(all_caps_word_own_line)
    df["Poem"] = df["Poem"].apply(roman_numeral_own_line)
    # "I." mid-line → "I" (first-person)
    df["Poem"] = df["Poem"].fillna("").apply(lambda t: I_PERIOD_MIDLINE_RE.sub(r"\1I\2", t) if isinstance(t, str) else t)

    # Pre-1800 only: split concatenated words (two uppercase letters not adjacent)
    year = pd.to_numeric(df["Year"], errors="coerce")
    pre1800 = year < 1800
    if pre1800.any():
        df.loc[pre1800, "Poem"] = df.loc[pre1800, "Poem"].apply(pre1800_uppercase_splits)
        df.loc[pre1800, "Poem"] = df.loc[pre1800, "Poem"].apply(pre1800_break_before_cap_conjunction)
        df.loc[pre1800, "Poem"] = df.loc[pre1800, "Poem"].apply(pre1800_punct_space_cap_to_newline)
        df.loc[pre1800, "Poem"] = df.loc[pre1800, "Poem"].apply(pre1800_merge_list_like_lines)
        print(f"Applied pre-1800 rules (incl. list-like merge) to {pre1800.sum()} poems.")

    # Pre-1900 only: split on multiple spaces
    pre1900 = year < 1900
    if pre1900.any():
        df.loc[pre1900, "Poem"] = df.loc[pre1900, "Poem"].apply(spaces_to_newlines)
        print(f"Applied pre-1900 multiple-spaces split to {pre1900.sum()} poems.")

    df.to_csv(CSV_PATH, index=False)
    print(f"Normalized Poem column in {CSV_PATH}.")


if __name__ == "__main__":
    main()
