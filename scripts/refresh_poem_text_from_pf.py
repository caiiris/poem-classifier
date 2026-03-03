"""
Refresh poem text from Poetry Foundation for pre-1800 poems that still need fixing (Track A).
Only scrapes poems that have "punctuation, space, capitalized word" (e.g. "food, And") —
those likely lost a line break; skip others that are already fine.

Usage:
  pip install playwright beautifulsoup4 && playwright install chromium
  python scripts/refresh_poem_text_from_pf.py [--delay 0.5] [--limit N]

Resume-safe: cache at data/refresh_poem_text_cache.csv (Title, Poet, Poem).
Reads and updates: data/PoetryFoundationData_with_year.csv
"""

import re
import time
from pathlib import Path
from urllib.parse import quote_plus, urljoin

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Only scrape poems with this pattern: punctuation, space(s), then capital letter (e.g. "food, And")
PUNCT_SPACE_CAP_RE = re.compile(r"[.,;:!?]\s+[A-Z]")


def needs_punct_space_fix(poem: str) -> bool:
    """True if poem has punctuation then space then capitalized word (likely lost line break)."""
    if not poem or not isinstance(poem, str):
        return False
    return bool(PUNCT_SPACE_CAP_RE.search(poem))
WITH_YEAR_CSV = DATA_DIR / "PoetryFoundationData_with_year.csv"
CACHE_CSV = DATA_DIR / "refresh_poem_text_cache.csv"
BASE_URL = "https://www.poetryfoundation.org"
SEARCH_URL = BASE_URL + "/search"


def extract_poem_body_from_page_text(text: str) -> str | None:
    """Extract poem body from full page innerText. Drops header/nav and Copyright/Source/Notes."""
    if not text or len(text.strip()) < 20:
        return None
    # Take everything before Copyright / Source / Notes (PF puts poem first, then metadata)
    parts = re.split(
        r"\s*(?:Poem copyright|Copyright\s*©|Source\s*:|Notes\s*:)\s*",
        text,
        maxsplit=1,
        flags=re.I,
    )
    block = parts[0].strip()
    if not block:
        return None
    # Drop leading lines that look like nav/site (e.g. "Poetry Foundation", "Poems", single-word lines)
    lines = block.split("\n")
    start = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            start = i + 1
            continue
        # Skip short lines that are likely title/poet or nav (first 2 non-empty often title + poet)
        if i - start < 3 and len(line) < 80 and line.lower() in ("poetry foundation", "poems", "poets"):
            start = i + 1
            continue
        # Keep the rest (poem body)
        break
    body = "\n".join(lines[start:]).strip()
    return body if len(body) > 20 else None


def search_poem_url_playwright(page, title: str, poet: str) -> str | None:
    """Search PF and return first poem page URL."""
    query = f"{title} {poet}".strip()
    if not query:
        return None
    url = f"{SEARCH_URL}?query={quote_plus(query)}&refinement=poems"
    try:
        page.goto(url, wait_until="networkidle", timeout=20000)
        time.sleep(0.3)
        content = page.content()
    except Exception:
        return None
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "/poems/" in href or "/poem/" in href:
            full = urljoin(BASE_URL, href)
            if "search" not in full:
                return full
    return None


def fetch_poem_body_playwright(page, url: str, delay: float) -> str | None:
    """Load poem page and return extracted poem body."""
    try:
        page.goto(url, wait_until="networkidle", timeout=20000)
        time.sleep(delay)
        text = page.evaluate("() => document.body.innerText") or ""
        return extract_poem_body_from_page_text(text)
    except Exception:
        return None


def main():
    import argparse
    p = argparse.ArgumentParser(description="Refresh pre-1800 poem text from Poetry Foundation.")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Seconds between requests (default 0.5; increase if PF slows or blocks).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N pre-1800 poems (for testing).")
    args = p.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright required. Install with:")
        print("  pip install playwright beautifulsoup4")
        print("  playwright install chromium")
        return

    df = pd.read_csv(WITH_YEAR_CSV)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    pre1800 = df["Year"] < 1800
    # Only poems that have "punctuation, space, capitalized word" (e.g. "food, And")
    need_fix = df["Poem"].fillna("").apply(needs_punct_space_fix)
    subset = df[pre1800 & need_fix].copy()
    if args.limit:
        subset = subset.head(args.limit)
    total = len(subset)
    n_pre1800 = pre1800.sum()
    n_need_fix = (pre1800 & need_fix).sum()
    if total == 0:
        print(f"No pre-1800 poems with 'punct space cap' pattern to process (of {n_pre1800} pre-1800, {n_need_fix} need fix).")
        return
    print(f"Pre-1800 poems with 'punct space cap' pattern: {n_need_fix} (of {n_pre1800} pre-1800). Processing {total}.")

    # Load cache: (Title, Poet) -> Poem
    cache: dict[tuple[str, str], str] = {}
    if CACHE_CSV.exists():
        try:
            c = pd.read_csv(CACHE_CSV)
            for _, row in c.iterrows():
                key = (str(row["Title"]).strip(), str(row["Poet"]).strip())
                poem = str(row["Poem"]).strip() if pd.notna(row.get("Poem")) else ""
                if poem:
                    cache[key] = poem
            print(f"Resuming: {len(cache)} poems already in cache.")
        except Exception:
            pass

    delay = max(0.2, args.delay)
    print(f"Processing {total} pre-1800 poems (delay={delay}s). Increase --delay if PF slows.\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(user_agent="PoetryEraProject/1.0 (educational)")
        page = context.new_page()
        try:
            for i, (idx, row) in enumerate(subset.iterrows()):
                title = str(row["Title"]).strip()
                poet = str(row["Poet"]).strip()
                if not title or not poet or poet == "nan":
                    continue
                key = (title, poet)
                if key in cache:
                    continue
                time.sleep(delay)
                url = search_poem_url_playwright(page, title, poet)
                if not url:
                    continue
                time.sleep(delay)
                body = fetch_poem_body_playwright(page, url, delay)
                if body:
                    cache[key] = body
                if (i + 1) % 50 == 0 or (i + 1) == total:
                    print(f"  Progress: {len(cache)} poems fetched...")
                    # Persist cache
                    pd.DataFrame(
                        [{"Title": k[0], "Poet": k[1], "Poem": v} for k, v in cache.items()]
                    ).to_csv(CACHE_CSV, index=False)
        finally:
            browser.close()

    # Write cache
    if cache:
        pd.DataFrame(
            [{"Title": k[0], "Poet": k[1], "Poem": v} for k, v in cache.items()]
        ).to_csv(CACHE_CSV, index=False)
        print(f"Cache saved: {len(cache)} poems.")

    # Update main CSV: set Poem from cache for pre-1800 rows
    def updated_poem(row):
        key = (str(row["Title"]).strip(), str(row["Poet"]).strip())
        return cache.get(key, row["Poem"])

    df.loc[pre1800, "Poem"] = df.loc[pre1800].apply(updated_poem, axis=1)
    df.to_csv(WITH_YEAR_CSV, index=False)
    print(f"Updated {WITH_YEAR_CSV} with PF text for {len(cache)} pre-1800 poems.")
    print("Re-run normalize_poem_line_breaks.py then plot_all_features_log.py to refresh features/plots.")


if __name__ == "__main__":
    main()
