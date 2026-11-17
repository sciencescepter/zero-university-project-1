"""Stage 4: line up the pipeline's news decisions against the labelled test set.

Reads a before run and an after run (outputs/stage4/*_news_decisions.json),
joins each one to the test set with pandas, and pulls out every news call
that is stricter than the true label (keep < age-restrict < remove).

It also looks up what each run cost in LangSmith (the cost OpenRouter
reported on every traced batch) and divides by the number of clips.

Writes outputs/stage4/stricter_review.md and prints the same numbers.
"""

import json
import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from langsmith import Client

from call_model import PROJECT_ROOT
from load_clip import CLIP_TITLE, load_dataset

STAGE4 = PROJECT_ROOT / "outputs" / "stage4"
RUNS = {"before": STAGE4 / "before_news_decisions.json", "after": STAGE4 / "after_news_decisions.json"}
REVIEW_PATH = STAGE4 / "stricter_review.md"
STRICTNESS = {"keep": 0, "age-restrict": 1, "remove": 2}


def line_up(decisions_path) -> tuple[pd.DataFrame, dict]:
    """One row per news clip: the pipeline's action next to the true label."""
    saved = json.loads(decisions_path.read_text())
    rows = [
        {
            "clip_id": clip["clip_id"],
            "action": clip["decision"]["action"] if clip["status"] == "decision" else None,
            "reason_code": clip["decision"]["reason_code"] if clip["status"] == "decision" else None,
        }
        for clip in saved["clips"]
    ]
    truth = load_dataset()[["clip_id", "title", "channel_type", "ground_truth_label", "ground_truth_policy_reason"]]
    df = pd.DataFrame(rows).merge(truth, on="clip_id", how="left", validate="one_to_one")
    df["direction"] = (df["action"].map(STRICTNESS) - df["ground_truth_label"].map(STRICTNESS)).map(
        lambda gap: "failed" if pd.isna(gap) else "stricter" if gap > 0 else "looser" if gap < 0 else "match"
    )
    return df, saved


def run_cost(saved: dict, clip_ids: set[str]) -> tuple[float, int]:
    """Total cost and batch count of the traced batches belonging to one run.

    A run's batches are the triage_batch traces that started in the 15 minutes
    before the run saved its file and that together cover exactly its clips.
    """
    finished = datetime.fromisoformat(saved["run_at_utc"])
    client = Client()
    candidates = sorted(
        (
            run
            for run in client.list_runs(
                project_name=os.environ["LANGSMITH_PROJECT"],
                run_type="chain",
                filter='eq(name, "triage_batch")',
                start_time=finished - timedelta(minutes=15),
            )
            if run.start_time.replace(tzinfo=finished.tzinfo) <= finished
            and set((run.inputs or {}).get("clip_ids", [])) <= clip_ids
        ),
        key=lambda run: run.start_time,
        reverse=True,
    )
    # Walk back from the newest batch, so earlier test calls on the same clips are left out.
    batches, covered = [], []
    for run in candidates:
        ids = run.inputs["clip_ids"]
        if set(ids) & set(covered):
            break
        batches.append(run)
        covered += ids
        if len(covered) == len(clip_ids):
            break
    if sorted(covered) != sorted(clip_ids):
        raise ValueError(f"traced batches cover {len(covered)} clips, expected {len(clip_ids)}")
    return sum(float(run.total_cost or 0) for run in batches), len(batches)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    results = {}
    for name, path in RUNS.items():
        df, saved = line_up(path)
        cost, batch_count = run_cost(saved, set(df["clip_id"]))
        ana = df[df["title"] == CLIP_TITLE].iloc[0]
        results[name] = {
            "df": df,
            "run_at": saved["run_at_utc"],
            "counts": df["direction"].value_counts().reindex(["match", "stricter", "looser", "failed"], fill_value=0),
            "ana": f"{ana['action']} ({ana['reason_code']})",
            "cost": cost,
            "batches": batch_count,
            "per_clip": cost / len(df),
        }

    before, after = results["before"], results["after"]
    lines = [
        "# Stage 4: news calls stricter than the truth",
        "",
        "Made by `src/review_stricter.py`. Each run's 64 news decisions are joined to the labelled test set",
        "with pandas. A call is *stricter* when the pipeline's action is more severe than the true label",
        "(keep < age-restrict < remove). Costs are the OpenRouter costs on each run's traced batches in LangSmith.",
        "",
        "## Before and after",
        "",
        "| | Before | After |",
        "|---|---|---|",
        f"| Run finished (UTC) | {before['run_at']} | {after['run_at']} |",
        f"| News clips | {len(before['df'])} | {len(after['df'])} |",
        f"| **Stricter than truth** | **{before['counts']['stricter']}** | **{after['counts']['stricter']}** |",
        f"| Looser than truth | {before['counts']['looser']} | {after['counts']['looser']} |",
        f"| Match | {before['counts']['match']} | {after['counts']['match']} |",
        f"| Failed | {before['counts']['failed']} | {after['counts']['failed']} |",
        f"| Ana's clip (yt_c0143, truth age-restrict) | {before['ana']} | {after['ana']} |",
        f"| Total cost ({before['batches']} / {after['batches']} batches) | ${before['cost']:.4f} | ${after['cost']:.4f} |",
        f"| Cost per clip | ${before['per_clip']:.6f} | ${after['per_clip']:.6f} |",
        "",
        "## Review table: stricter calls",
        "",
        "| Run | clip_id | Pipeline | Truth | Trace finding | Cause |",
        "|---|---|---|---|---|---|",
    ]
    for name in RUNS:
        stricter = results[name]["df"][results[name]["df"]["direction"] == "stricter"]
        for _, row in stricter.iterrows():
            lines.append(
                f"| {name} | {row['clip_id']} | {row['action']} | {row['ground_truth_label']} | (read trace) | (to fill) |"
            )
        if stricter.empty:
            lines.append(f"| {name} | none | | | No stricter calls, so no trace to open | No cause to fix |")

    lines += ["", "## Mismatches for reference (looser than truth)", "", "| Run | clip_id | Pipeline | Truth |", "|---|---|---|---|"]
    for name in RUNS:
        for _, row in results[name]["df"][results[name]["df"]["direction"] == "looser"].iterrows():
            lines.append(f"| {name} | {row['clip_id']} | {row['action']} ({row['reason_code']}) | {row['ground_truth_label']} |")

    REVIEW_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[6:20]))
    print(f"\nReview written to {REVIEW_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
