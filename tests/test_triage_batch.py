"""Offline tests for triage_batch's prompt and reply handling. No model is called."""

import json

import pytest

from load_clip import CLIP_TITLE, load_dataset
from triage_batch import MAX_BATCH_SIZE, build_batch_prompt, parse_batch_reply, triage_batch

DF = load_dataset()
ANA = DF[DF["title"] == CLIP_TITLE].iloc[0].to_dict()  # yt_c0143, 198 seconds
SHORT = DF[DF["duration_seconds"] < 100].iloc[0].to_dict()  # a clip shorter than 100 seconds

NEWS_REASON = "Graphic violence shown in a news or documentary context"


def decision_for(clip_id: str, end: float) -> dict:
    return {
        "clip_id": clip_id,
        "action": "age-restrict",
        "policy_reason": NEWS_REASON,
        "reason_code": "VC-03",
        "evidence": [{"start": 16, "end": end, "description": "A bottle is thrown at officers."}],
    }


def test_prompt_uses_only_title_transcript_and_captions():
    prompt = build_batch_prompt([ANA, SHORT])

    for clip in (ANA, SHORT):
        assert clip["clip_id"] in prompt
        assert clip["title"] in prompt
        assert clip["transcript_timestamped"] in prompt
        assert clip["frame_captions_timestamped"] in prompt
        # The answer key must never be sent.
        assert str(clip["ground_truth_policy_reason"]) not in prompt
        assert "violent_segments" not in prompt
    assert "ground_truth" not in prompt


def test_valid_reply_gives_one_decision_per_clip_id():
    reply = json.dumps([decision_for(ANA["clip_id"], 55), decision_for(SHORT["clip_id"], 55)])

    decisions, errors = parse_batch_reply(reply, [ANA, SHORT])

    assert errors == {}
    assert set(decisions) == {ANA["clip_id"], SHORT["clip_id"]}
    assert decisions[ANA["clip_id"]].action == "age-restrict"


def test_each_decision_is_checked_against_its_own_clips_duration():
    """The same 16-150s moment fits Ana's 198s clip but runs past the short clip."""
    assert SHORT["duration_seconds"] < 150 < ANA["duration_seconds"]
    reply = json.dumps([decision_for(ANA["clip_id"], 150), decision_for(SHORT["clip_id"], 150)])

    decisions, errors = parse_batch_reply(reply, [ANA, SHORT])

    assert set(decisions) == {ANA["clip_id"]}
    assert set(errors) == {SHORT["clip_id"]}
    assert f"runs past the clip's end ({float(SHORT['duration_seconds'])}s)" in errors[SHORT["clip_id"]]


def test_broken_replies_are_reported_not_defaulted():
    # Not JSON at all: every clip is reported, none gets a decision.
    decisions, errors = parse_batch_reply("Label: age-restrict", [ANA, SHORT])
    assert decisions == {}
    assert set(errors) == {ANA["clip_id"], SHORT["clip_id"]}
    assert all("not valid JSON" in reason for reason in errors.values())

    # One clip missing, one unknown clip_id, one refused label.
    reply = json.dumps(
        [
            {**decision_for(ANA["clip_id"], 55), "action": "flag"},
            decision_for("yt_c9999", 55),
        ]
    )
    decisions, errors = parse_batch_reply(reply, [ANA, SHORT])
    assert decisions == {}
    assert "literal_error" in errors[ANA["clip_id"]]
    assert errors["yt_c9999"] == "decision is for a clip_id that was not sent"
    assert errors[SHORT["clip_id"]] == "no decision returned for this clip_id"


def test_batch_size_is_limited_before_any_request():
    too_many = DF.head(MAX_BATCH_SIZE + 1)
    with pytest.raises(ValueError, match="1 to 10 clips, got 11"):
        triage_batch(too_many)
    with pytest.raises(ValueError, match="1 to 10 clips, got 0"):
        triage_batch([])
