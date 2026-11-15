"""Triage a batch of up to 10 clips in one OpenRouter request.

triage_batch(clips) sends one prompt for the whole batch, asks for a JSON
array of decisions, and validates each decision on its own through the
decision schema, using that clip's duration_seconds from the test set.

A broken reply is retried once with its errors attached. A second broken
reply raises TriageBatchError. There is no default label.

Every batch request is traced to LangSmith, with the token usage (and cost)
OpenRouter reports attached to the traced model run.

The model id comes from config.yaml and the keys from .env. Neither is in
this file.
"""

import json
import re
from collections.abc import Mapping, Sequence

from dotenv import load_dotenv
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from langsmith.run_trees import get_cached_client
from openai import OpenAI
from pydantic import ValidationError

from call_model import OPENROUTER_BASE_URL, PROJECT_ROOT, load_settings
from decision_schema import APPROVED_REASONS, Decision, validate_decision

# Load .env before any traced call, so LangSmith sees LANGSMITH_TRACING,
# LANGSMITH_API_KEY and LANGSMITH_PROJECT.
load_dotenv(PROJECT_ROOT / ".env")

MAX_BATCH_SIZE = 10
# Room for 10 decisions. Also keeps OpenRouter from reserving credit for the
# model's full maximum reply.
MAX_REPLY_TOKENS = 6000
# The first request, plus exactly one retry that carries the errors back.
MAX_ATTEMPTS = 2


class TriageBatchError(Exception):
    """Raised when any clip in a batch has no valid decision. Never silently skipped.

    .decisions holds the clips that did validate; .errors maps each failed
    clip_id to the reason it failed.
    """

    def __init__(
        self,
        decisions: dict[str, Decision],
        errors: dict[str, str],
        first_attempt_errors: dict[str, str] | None = None,
    ):
        self.decisions = decisions
        self.errors = errors
        self.first_attempt_errors = first_attempt_errors or {}
        details = "\n".join(f"  {clip_id}: {reason}" for clip_id, reason in errors.items())
        super().__init__(
            f"{len(errors)} clip(s) still have no valid decision after {MAX_ATTEMPTS} attempts:\n{details}"
        )


def _as_records(clips) -> list[dict]:
    """Accept a DataFrame of test set rows, or a list of row dicts / Series."""
    if hasattr(clips, "to_dict") and hasattr(clips, "columns"):
        return clips.to_dict("records")
    return [dict(clip) for clip in clips]


def build_batch_prompt(clips: Sequence[Mapping]) -> str:
    """Build one prompt for the batch from title, transcript and frame captions only.

    The ground_truth columns are never read here, so the answers can't leak.
    """
    reasons = "\n".join(
        f'- {code}: "{wording}" (only with action: {", ".join(sorted(actions))})'
        for wording, (code, actions) in APPROVED_REASONS.items()
    )

    clip_blocks = []
    for clip in clips:
        clip_blocks.append(
            f"clip_id: {clip['clip_id']}\n"
            f"Title: {clip['title']}\n"
            f"Transcript (timestamped): {clip['transcript_timestamped']}\n"
            f"Frame captions (timestamped): {clip['frame_captions_timestamped']}"
        )
    clip_text = "\n\n---\n\n".join(clip_blocks)

    return f"""You are reviewing YouTube clips for a violent content moderation team.

Give one decision for each of the {len(clips)} clips below.

Actions (choose exactly one per clip, written exactly like this):
- keep: stays up with no restriction. Violence is absent, incidental or non-graphic.
- age-restrict: stays up for adults only. Graphic violence with news, documentary, educational or artistic context.
- remove: taken down. Violence posted to glorify, incite or threaten, or graphic violence with no context.

Approved policy reasons (copy the wording exactly; each only goes with the action shown):
{reasons}

Evidence moments:
- age-restrict and remove need at least one moment. keep may have none.
- start and end are numbers of SECONDS from the start of the clip (convert mm:ss, so 01:20 is 80).
- start must be less than end, and both must be inside the clip.
- description is one plain sentence saying what is shown or said.

Reply with ONLY a JSON array, with no other text and no code fences. It must hold exactly one object per clip_id, each shaped like this:
{{
  "clip_id": "<the clip_id>",
  "action": "keep" | "age-restrict" | "remove",
  "policy_reason": "<approved wording, exactly>",
  "reason_code": "<matching code, e.g. VC-03>",
  "evidence": [{{"start": <seconds>, "end": <seconds>, "description": "<one sentence>"}}]
}}

Clips:

{clip_text}"""


def _strip_code_fences(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, re.DOTALL)
    return match.group(1) if match else text.strip()


def parse_batch_reply(
    reply_text: str, clips: Sequence[Mapping]
) -> tuple[dict[str, Decision], dict[str, str]]:
    """Turn the model's reply into validated decisions, one clip at a time.

    Returns (decisions, errors). Each decision is validated on its own with
    its own clip's duration_seconds. A clip with a missing, duplicated or
    invalid decision gets an entry in errors. Nothing is defaulted.
    """
    durations = {clip["clip_id"]: float(clip["duration_seconds"]) for clip in clips}
    decisions: dict[str, Decision] = {}
    errors: dict[str, str] = {}

    try:
        items = json.loads(_strip_code_fences(reply_text))
    except json.JSONDecodeError as error:
        return {}, {clip_id: f"reply is not valid JSON: {error}" for clip_id in durations}
    if not isinstance(items, list):
        return {}, {clip_id: "reply is not a JSON array of decisions" for clip_id in durations}

    seen: set[str] = set()
    for position, item in enumerate(items):
        if not isinstance(item, dict) or "clip_id" not in item:
            errors[f"<item {position}>"] = "array item is not an object with a clip_id"
            continue

        fields = dict(item)
        clip_id = str(fields.pop("clip_id"))
        if clip_id not in durations:
            errors[clip_id] = "decision is for a clip_id that was not sent"
            continue
        if clip_id in seen:
            decisions.pop(clip_id, None)
            errors[clip_id] = "more than one decision returned for this clip_id"
            continue
        seen.add(clip_id)

        try:
            decisions[clip_id] = validate_decision(fields, durations[clip_id])
        except ValidationError as error:
            errors[clip_id] = str(error)

    for clip_id in durations:
        if clip_id not in seen:
            errors[clip_id] = "no decision returned for this clip_id"

    return decisions, errors


def build_retry_message(errors: Mapping[str, str], clip_count: int) -> str:
    """The follow-up sent on the single retry: every error, then the same ask again."""
    problems = "\n".join(f"- {clip_id}: {reason}" for clip_id, reason in errors.items())
    return (
        "Your reply could not be used. These problems were found:\n"
        f"{problems}\n\n"
        f"Reply again with ONLY the corrected JSON array, holding one decision for each of "
        f"the {clip_count} clip_id(s) sent, in the shape described before. No other text. "
        "Every rule in the first message still applies, not only the ones listed above."
    )


def _usage_metadata(usage) -> dict:
    """Convert the usage block OpenRouter sends back into LangSmith's format."""
    metadata = {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None)
    if reasoning_tokens:
        metadata["output_token_details"] = {"reasoning": reasoning_tokens}
    cost = getattr(usage, "cost", None)  # OpenRouter's own charge for the request, in USD
    if cost is not None:
        metadata["total_cost"] = float(cost)
    return metadata


@traceable(run_type="llm", name="openrouter_chat")
def _call_openrouter(messages: list[dict], model_id: str, api_key: str) -> dict:
    """One traced model request. Records OpenRouter's token usage on the run."""
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=MAX_REPLY_TOKENS,
        temperature=0,
        extra_body={"usage": {"include": True}},  # ask OpenRouter to report cost too
    )

    run = get_current_run_tree()
    if run is not None and response.usage is not None:
        run.set(usage_metadata=_usage_metadata(response.usage))

    return {"role": "assistant", "content": response.choices[0].message.content or ""}


def _trace_inputs(inputs: dict) -> dict:
    """Log only the clip ids on the batch run, not whole rows with ground truth."""
    clips = _as_records(inputs.get("clips", []))
    return {"clip_ids": [clip.get("clip_id") for clip in clips]}


def _trace_outputs(decisions) -> dict:
    if not isinstance(decisions, dict):  # nothing was returned because the batch raised
        return {}
    return {clip_id: decision.model_dump() for clip_id, decision in decisions.items()}


@traceable(
    run_type="chain",
    name="triage_batch",
    process_inputs=_trace_inputs,
    process_outputs=_trace_outputs,
)
def triage_batch(clips) -> dict[str, Decision]:
    """Triage up to 10 clips in one request. Returns {clip_id: Decision}.

    If the reply is broken (not JSON, a clip missing, or any decision the
    schema rejects), the model is asked once more, with the errors attached.
    If that retry is broken too, TriageBatchError is raised. No clip is ever
    given a default label.

    Raises ValueError for an empty or oversized batch.
    """
    records = _as_records(clips)
    if not 1 <= len(records) <= MAX_BATCH_SIZE:
        raise ValueError(f"a batch must hold 1 to {MAX_BATCH_SIZE} clips, got {len(records)}")
    clip_ids = [record["clip_id"] for record in records]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("a batch must not contain the same clip_id twice")

    api_key, model_id = load_settings()
    messages = [{"role": "user", "content": build_batch_prompt(records)}]
    first_attempt_errors: dict[str, str] = {}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply = _call_openrouter(
            messages,
            model_id,
            api_key,
            langsmith_extra={
                "metadata": {
                    "ls_provider": "openrouter",
                    "ls_model_name": model_id,
                    "attempt": attempt,
                }
            },
        )
        decisions, errors = parse_batch_reply(reply["content"], records)
        if not errors:
            return decisions

        if attempt == 1:
            # Retry once: show the model its own reply and exactly what was wrong with it.
            first_attempt_errors = errors
            messages = [
                *messages,
                {"role": "assistant", "content": reply["content"]},
                {"role": "user", "content": build_retry_message(errors, len(records))},
            ]

    # The retry failed too. Raise loudly; never fall back to a default label.
    raise TriageBatchError(decisions, errors, first_attempt_errors)


def flush_traces() -> None:
    """Send any traces still queued before the program exits."""
    get_cached_client().flush()


if __name__ == "__main__":
    from load_clip import CLIP_TITLE, load_dataset

    df = load_dataset()
    ana = df[df["title"] == CLIP_TITLE]
    others = df[df["is_news"] & (df["title"] != CLIP_TITLE)].head(2)
    batch = [*ana.to_dict("records"), *others.to_dict("records")]

    try:
        results = triage_batch(batch)
        for clip_id, decision in results.items():
            spans = ", ".join(f"{m.start:g}-{m.end:g}s" for m in decision.evidence) or "no evidence"
            print(f"{clip_id}: {decision.action} | {decision.policy_reason} | {spans}")
    finally:
        flush_traces()
