"""Tests for the decision schema, using Ana's real model reply where possible.

Every rejection test is pinned to pydantic's ValidationError and then checks
WHICH field failed and WHY, so a test can't go green because of some
unrelated error.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from decision_schema import parse_raw_reply, validate_decision, validate_decision_json
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


def rejection(data: dict) -> list[tuple[tuple, str, str]]:
    """Validate data, require a schema rejection, and return (where, kind, message) per error."""
    with pytest.raises(ValidationError) as caught:
        validate_decision(data, CLIP_DURATION)
    return [(e["loc"], e["type"], e["msg"]) for e in caught.value.errors()]


def test_ana_raw_reply_is_rejected():
    """The real reply has no policy reason and only point timestamps, so it can't be enforced."""
    parsed = parse_raw_reply(RAW_REPLY)
    assert parsed["action"] == "age-restrict"
    assert [(m["start"], m["end"]) for m in parsed["evidence"]] == [(30, 30), (110, 110)]

    errors = rejection(parsed)

    assert (("policy_reason",), "missing") in [(loc, kind) for loc, kind, _ in errors]
    for index in (0, 1):
        assert any(
            loc == ("evidence", index) and "must start before it ends" in msg
            for loc, _, msg in errors
        )
    assert len(errors) == 3


@pytest.mark.parametrize("refused_action", ["flag", "restricted", "review"])
def test_refused_action_is_rejected(refused_action):
    """The policy's own refused examples must be stopped by the action field itself."""
    data = ana_reply_in_schema_form()
    data["action"] = refused_action

    errors = rejection(data)

    # Exactly one error, on the action field, of the "not one of the allowed values" kind.
    assert [(loc, kind) for loc, kind, _ in errors] == [(("action",), "literal_error")]


def test_moment_that_runs_backwards_is_rejected():
    # Backwards (end before start), and zero-length (start equal to end).
    for start, end in [(55, 16), (30, 30)]:
        data = ana_reply_in_schema_form()
        data["evidence"][0]["start"] = start
        data["evidence"][0]["end"] = end

        errors = rejection(data)

        assert len(errors) == 1
        loc, _, msg = errors[0]
        assert loc == ("evidence", 0)
        assert "must start before it ends" in msg


def test_moment_past_clip_end_is_rejected():
    data = ana_reply_in_schema_form()
    data["evidence"][1]["end"] = CLIP_DURATION + 1  # 199s on a 198s clip

    errors = rejection(data)

    assert len(errors) == 1
    loc, _, msg = errors[0]
    assert loc == ("evidence", 1)
    assert "runs past the clip's end (198.0s)" in msg

    # A runtime supplied by the model is refused outright, so it can't override ours.
    data = ana_reply_in_schema_form()
    data["clip_duration"] = 999
    errors = rejection(data)
    assert [(loc, kind) for loc, kind, _ in errors] == [(("clip_duration",), "extra_forbidden")]


def test_valid_decision_survives_json_round_trip():
    """A valid decision, saved as JSON and loaded back with the same runtime, is unchanged."""
    decision = validate_decision(ana_reply_in_schema_form(), CLIP_DURATION)
    assert decision.action == "age-restrict"
    assert decision.policy_reason == NEWS_REASON
    assert len(decision.evidence) == 2

    json_text = decision.model_dump_json()
    assert isinstance(json_text, str)
    assert "duration" not in json_text  # the runtime is never stored in the decision

    rebuilt = validate_decision_json(json_text, CLIP_DURATION)

    assert rebuilt == decision
    assert json.loads(rebuilt.model_dump_json()) == json.loads(json_text)
