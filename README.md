"""
Data Engineer Job Pipeline
- Scrapes LinkedIn, Indeed, Glassdoor, ZipRecruiter for Data Engineer jobs posted in last 24h
- Uses an LLM to filter for companies that offer visa sponsorship / accept OPT candidates
- Outputs a ranked table + saves results to CSV
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn

# ── JobSpy import (handles multi-site scraping) ──────────────────────────────
try:
    from jobspy import scrape_jobs
except ImportError:
    raise ImportError(
        "python-jobspy is not installed. Run: pip install -r requirements.txt"
    )

# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
console = Console()
logging.basicConfig(level=logging.WARNING)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROXY          = os.getenv("PROXY", None)

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Copy .env.example → .env and add your key."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SCRAPE
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_TERMS = [
    "Data Engineer",
    "Data Pipeline Engineer",
    "Analytics Engineer",
]

SITES = ["linkedin", "indeed", "glassdoor", "zip_recruiter"]

def scrape_all_sources(hours_old: int = 24) -> pd.DataFrame:
    """Scrape multiple job boards and return a deduplicated DataFrame."""
    all_jobs: list[pd.DataFrame] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        for term in SEARCH_TERMS:
            task = progress.add_task(f"Scraping: {term!r}...", total=None)
            try:
                df = scrape_jobs(
                    site_name=SITES,
                    search_term=term,
                    location="United States",
                    results_wanted=50,          # per site
                    hours_old=hours_old,
                    country_indeed="USA",
                    linkedin_fetch_description=True,
                    proxies=[PROXY] if PROXY else None,
                )
                if df is not None and not df.empty:
                    all_jobs.append(df)
                    console.log(f"[green]✓[/] {term!r}: {len(df)} jobs found")
                else:
                    console.log(f"[yellow]⚠[/] {term!r}: no results")
            except Exception as e:
                console.log(f"[red]✗[/] {term!r}: {e}")
            progress.remove_task(task)
            time.sleep(2)   # be polite between searches

    if not all_jobs:
        console.print("[bold red]No jobs scraped from any source.[/]")
        return pd.DataFrame()

    combined = pd.concat(all_jobs, ignore_index=True)

    # Deduplicate on (title, company, location)
    combined.drop_duplicates(
        subset=["title", "company", "location"], keep="first", inplace=True
    )
    combined.reset_index(drop=True, inplace=True)
    console.print(f"\n[bold cyan]Total unique jobs after dedup:[/] {len(combined)}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM FILTER — classify each job description
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a visa/work-authorization expert assistant helping international students in the US on OPT (Optional Practical Training) find jobs.

Given a job posting (title, company, and description), classify whether:
1. The company explicitly SPONSORS visas (H-1B, TN, etc.) OR does not exclude sponsored candidates
2. The company explicitly ACCEPTS or does not exclude OPT / CPT candidates
3. The company explicitly says "no sponsorship" or "US citizens/permanent residents only"

Return a JSON object with exactly these keys:
{
  "sponsors_visa": true | false | null,
  "accepts_opt": true | false | null,
  "confidence": "high" | "medium" | "low",
  "evidence": "<one sentence quoting or summarizing the relevant text, or 'No explicit mention'>"
}

Rules:
- sponsors_visa = true  → posting mentions sponsorship offered, or does not restrict to citizens/GC
- sponsors_visa = false → posting says "no sponsorship", "must be authorized", "US citizen or GC only"
- sponsors_visa = null  → no mention either way
- accepts_opt = true    → mentions OPT, CPT, F-1, international students welcome
- accepts_opt = false   → explicitly excludes OPT / requires GC or citizenship
- accepts_opt = null    → no explicit mention
- confidence = high when direct quotes exist; medium when inferred; low when description is missing
"""

def _build_user_message(row: pd.Series) -> str:
    description = str(row.get("description", "") or "").strip()
    if not description:
        description = "(no description available)"
    # Truncate to ~3000 chars to keep token cost low
    description = description[:3000]
    return (
        f"Job Title: {row.get('title', 'N/A')}\n"
        f"Company:   {row.get('company', 'N/A')}\n"
        f"Location:  {row.get('location', 'N/A')}\n\n"
        f"Description:\n{description}"
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)
def classify_job(row: pd.Series) -> dict:
    """Call the LLM to classify a single job posting."""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(row)},
        ],
        temperature=0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def classify_all(df: pd.DataFrame, batch_delay: float = 0.3) -> pd.DataFrame:
    """Classify every job and add result columns to the DataFrame."""
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Classifying {len(df)} jobs with LLM...", total=len(df)
        )
        for _, row in df.iterrows():
            try:
                classification = classify_job(row)
            except Exception as e:
                console.log(f"[red]LLM error for {row.get('company')}:[/] {e}")
                classification = {
                    "sponsors_visa": None,
                    "accepts_opt": None,
                    "confidence": "low",
                    "evidence": "LLM call failed",
                }
            results.append(classification)
            progress.advance(task)
            time.sleep(batch_delay)

    df["sponsors_visa"] = [r.get("sponsors_visa") for r in results]
    df["accepts_opt"]   = [r.get("accepts_opt")   for r in results]
    df["llm_confidence"]= [r.get("confidence")    for r in results]
    df["llm_evidence"]  = [r.get("evidence")       for r in results]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FILTER & RANK
# ─────────────────────────────────────────────────────────────────────────────
def filter_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep jobs where:
      - sponsors_visa is True OR not False  (i.e., true or null → still worth applying)
      - accepts_opt   is True OR not False
    Then rank: explicit positives first, unknowns second.
    """
    # Exclude only explicit rejections
    filtered = df[
        (df["sponsors_visa"] != False) &  # noqa: E712
        (df["accepts_opt"]   != False)    # noqa: E712
    ].copy()

    def rank_score(row):
        score = 0
        if row["sponsors_visa"] is True:  score += 4
        if row["accepts_opt"]   is True:  score += 4
        if row["sponsors_visa"] is None:  score += 1
        if row["accepts_opt"]   is None:  score += 1
        if row["llm_confidence"] == "high":   score += 2
        if row["llm_confidence"] == "medium": score += 1
        return score

    filtered["rank_score"] = filtered.apply(rank_score, axis=1)
    filtered.sort_values("rank_score", ascending=False, inplace=True)
    filtered.reset_index(drop=True, inplace=True)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# 4. OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
DISPLAY_COLS = [
    "title", "company", "location", "date_posted",
    "sponsors_visa", "accepts_opt", "llm_confidence", "llm_evidence",
    "job_url",
]

def display_results(df: pd.DataFrame) -> None:
    """Pretty-print results using Rich."""
    if df.empty:
        console.print("[bold red]No qualifying jobs found.[/]")
        return

    table = Table(
        title=f"Data Engineer Jobs — OPT/Sponsorship Friendly ({len(df)} results)",
        show_lines=True,
        highlight=True,
    )
    table.add_column("#",            style="dim",    width=3)
    table.add_column("Title",        style="bold",   max_width=30)
    table.add_column("Company",      style="cyan",   max_width=22)
    table.add_column("Location",     max_width=18)
    table.add_column("Posted",       max_width=12)
    table.add_column("Sponsors?",    max_width=9)
    table.add_column("OPT?",         max_width=7)
    table.add_column("Confidence",   max_width=10)
    table.add_column("Evidence",     max_width=40)

    for i, row in df.iterrows():
        def fmt_bool(val):
            if val is True:  return "[green]✓ Yes[/]"
            if val is False: return "[red]✗ No[/]"
            return "[yellow]?[/]"

        def fmt_conf(val):
            if val == "high":   return "[green]High[/]"
            if val == "medium": return "[yellow]Med[/]"
            return "[dim]Low[/]"

        posted = str(row.get("date_posted", ""))[:10]

        table.add_row(
            str(i + 1),
            str(row.get("title",   ""))[:30],
            str(row.get("company", ""))[:22],
            str(row.get("location",""))[:18],
            posted,
            fmt_bool(row.get("sponsors_visa")),
            fmt_bool(row.get("accepts_opt")),
            fmt_conf(row.get("llm_confidence")),
            str(row.get("llm_evidence", ""))[:40],
        )

    console.print(table)

    # Print URLs separately so they're clickable
    console.print("\n[bold]Job URLs:[/]")
    for i, row in df.iterrows():
        url = row.get("job_url", "")
        company = row.get("company", "")
        title   = row.get("title",   "")
        if url:
            console.print(f"  [dim]{i+1:>3}.[/] [cyan]{company}[/] — {title}\n       [link={url}]{url}[/link]")


def save_results(df: pd.DataFrame, output_dir: str = ".") -> str:
    """Save full + filtered results to CSV."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"opt_jobs_{ts}.csv"

    cols_to_save = [c for c in DISPLAY_COLS + ["rank_score"] if c in df.columns]
    df[cols_to_save].to_csv(out_path, index=False)
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(hours_old: int = 24, save_csv: bool = True) -> pd.DataFrame:
    console.rule("[bold blue]Step 1 — Scraping job boards")
    raw_df = scrape_all_sources(hours_old=hours_old)

    if raw_df.empty:
        console.print("[red]Pipeline stopped: no jobs to process.[/]")
        return pd.DataFrame()

    console.rule("[bold blue]Step 2 — LLM Classification (OPT / Sponsorship)")
    classified_df = classify_all(raw_df)

    console.rule("[bold blue]Step 3 — Filtering & Ranking")
    final_df = filter_and_rank(classified_df)
    console.print(
        f"[green]{len(final_df)}[/] jobs passed the filter "
        f"(removed {len(classified_df) - len(final_df)} explicit rejections)"
    )

    console.rule("[bold blue]Step 4 — Results")
    display_results(final_df)

    if save_csv and not final_df.empty:
        path = save_results(final_df)
        console.print(f"\n[bold green]✓ Saved to:[/] {path}")

    return final_df


if __name__ == "__main__":
    run_pipeline(hours_old=24, save_csv=True)
