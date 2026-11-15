"""Tests for the decision schema, using Ana's real model reply where possible."""

from pathlib import Path

import pytest

from decision_schema import parse_raw_reply, validate_decision
from load_clip import load_dataset, select_clip

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_REPLY = (PROJECT_ROOT / "outputs" / "ana_protest_raw_reply.txt").read_text()

# Runtime comes from the upload record (the dataset), never from the model.
CLIP = select_clip(load_dataset())
CLIP_DURATION = float(CLIP["duration_seconds"])  # 198.0 for yt_c0143

NEWS_REASON = "Graphic violence shown in a news or documentary context"


def ana_reply_in_schema_form() -> dict:
    """Ana's real reply, with the parts the policy requires filled in.

    Same action and same two moments. Each point timestamp is widened to the
    transcript segment it falls in, and the approved VC-03 reason is added.
    """
    return {
        "action": "age-restrict",
        "policy_reason": NEWS_REASON,
        "reason_code": "VC-03",
        "evidence": [
            {"start": 16, "end": 55, "description": "A bottle arcs through the air towards officers."},
            {"start": 100, "end": 140, "description": "A paramedic tends to an officer with a bleeding forehead."},
        ],
    }


def test_ana_raw_reply_is_rejected():
    """The real reply has no policy reason and only point timestamps, so it can't be enforced."""
    parsed = parse_raw_reply(RAW_REPLY)

    assert parsed["action"] == "age-restrict"
    assert [(m["start"], m["end"]) for m in parsed["evidence"]] == [(30, 30), (110, 110)]
    assert "policy_reason" not in parsed

    with pytest.raises(ValueError) as error:
        validate_decision(parsed, CLIP_DURATION)
    message = str(error.value)
    assert "policy_reason" in message
    assert "must start before it ends" in message


def test_ana_reply_in_schema_form_is_accepted():
    decision = validate_decision(ana_reply_in_schema_form(), CLIP_DURATION)

    assert decision.action == "age-restrict"
    assert decision.policy_reason == NEWS_REASON
    assert len(decision.evidence) == 2


def test_moment_that_runs_backwards_is_rejected():
    # Backwards (end before start), and zero-length (start equal to end).
    for start, end in [(55, 16), (30, 30)]:
        data = ana_reply_in_schema_form()
        data["evidence"][0]["start"] = start
        data["evidence"][0]["end"] = end

        with pytest.raises(ValueError, match="must start before it ends"):
            validate_decision(data, CLIP_DURATION)


def test_moment_past_clip_end_is_rejected():
    data = ana_reply_in_schema_form()
    data["evidence"][1]["end"] = CLIP_DURATION + 1  # 199s on a 198s clip

    with pytest.raises(ValueError, match="runs past the clip's end"):
        validate_decision(data, CLIP_DURATION)

    # A runtime supplied by the model is refused outright, so it can't override ours.
    data = ana_reply_in_schema_form()
    data["clip_duration"] = 999
    with pytest.raises(ValueError, match="clip_duration"):
        validate_decision(data, CLIP_DURATION)
