"""Run the triage pipeline on every news clip in the test set and check the labels.

Picks out the news clips (is_news), makes sure Ana's protest clip is among
them, and sends them through triage_batch in batches of 10. Prints one line
per clip. A clip with no valid decision is printed as FAILED, never skipped.
The decisions are saved to outputs/news_decisions.json.

The ground-truth label is only used here, after the model has replied.
"""

import json
from datetime import datetime, timezone

from call_model import PROJECT_ROOT, load_settings
from load_clip import CLIP_TITLE, load_dataset
from triage_batch import MAX_BATCH_SIZE, TriageBatchError, flush_traces, triage_batch

DECISIONS_PATH = PROJECT_ROOT / "outputs" / "news_decisions.json"


def select_news_clips() -> list[dict]:
    """All news clips as row dicts, with Ana's protest clip first."""
    df = load_dataset()
    news = df[df["is_news"]]
    ana = news[news["title"] == CLIP_TITLE]
    if len(ana) != 1:
        raise ValueError(f"Ana's protest clip must be among the news clips exactly once, found {len(ana)}")
    rest = news[news["title"] != CLIP_TITLE]
    return [*ana.to_dict("records"), *rest.to_dict("records")]


def run_batch(batch: list[dict]) -> tuple[dict, dict[str, str]]:
    """Return (decisions, errors) for one batch. Every clip lands in one or the other."""
    try:
        return triage_batch(batch), {}
    except TriageBatchError as error:
        errors = {
            clip["clip_id"]: error.errors.get(clip["clip_id"], "batch failed validation")
            for clip in batch
            if clip["clip_id"] not in error.decisions
        }
        return error.decisions, errors
    except Exception as error:  # e.g. the request itself failed
        reason = f"{type(error).__name__}: {error}"
        return {}, {clip["clip_id"]: reason for clip in batch}


def main() -> None:
    _, model_id = load_settings()
    clips = select_news_clips()
    batches = [clips[i : i + MAX_BATCH_SIZE] for i in range(0, len(clips), MAX_BATCH_SIZE)]
    print(f"Model: {model_id}")
    print(f"News clips: {len(clips)} in {len(batches)} batches of up to {MAX_BATCH_SIZE}")
    print(f"Ana's protest clip included: {clips[0]['clip_id']}\n")

    results = []
    for number, batch in enumerate(batches, start=1):
        print(f"--- Batch {number} of {len(batches)} ({len(batch)} clips)", flush=True)
        decisions, errors = run_batch(batch)

        for clip in batch:
            clip_id = clip["clip_id"]
            truth = clip["ground_truth_label"]
            record = {"clip_id": clip_id, "title": clip["title"], "ground_truth_label": truth}

            if clip_id in decisions:
                decision = decisions[clip_id]
                matches = decision.action == truth
                spans = ", ".join(f"{m.start:g}-{m.end:g}s" for m in decision.evidence) or "no evidence"
                print(
                    f"{clip_id} | {decision.action:<12} | truth {truth:<12} | "
                    f"{'MATCH   ' if matches else 'MISMATCH'} | {decision.reason_code or '-'} | {spans}",
                    flush=True,
                )
                record.update(status="decision", label_matches=matches, decision=decision.model_dump())
            else:
                reason = errors.get(clip_id, "no decision and no error recorded")
                first_line = " ".join(reason.split())[:160]
                print(f"{clip_id} | FAILED       | truth {truth:<12} | {first_line}", flush=True)
                record.update(status="failed", label_matches=None, error=reason)

            results.append(record)

    decided = [r for r in results if r["status"] == "decision"]
    failed = [r for r in results if r["status"] == "failed"]
    matched = [r for r in decided if r["label_matches"]]
    summary = {
        "clips": len(results),
        "decisions": len(decided),
        "failed": len(failed),
        "label_matches": len(matched),
        "label_mismatches": len(decided) - len(matched),
    }

    print(f"\nClips: {summary['clips']} | decisions: {summary['decisions']} | failed: {summary['failed']}")
    print(
        f"Label matches ground truth: {summary['label_matches']} of {summary['clips']} clips "
        f"({summary['label_matches'] / summary['clips']:.0%}); failed clips count as not matched"
    )

    DECISIONS_PATH.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "summary": summary,
                "clips": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Decisions saved to {DECISIONS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    finally:
        flush_traces()
