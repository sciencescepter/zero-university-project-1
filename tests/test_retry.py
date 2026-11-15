"""Offline tests for triage_batch's single retry. A fake stands in for the model.

The broken reply used here is Ana's real free-text reply from stage 1.
"""

import json
from pathlib import Path

import pytest

import triage_batch as tb
from load_clip import CLIP_TITLE, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANA_RAW_REPLY = (PROJECT_ROOT / "outputs" / "ana_protest_raw_reply.txt").read_text()

DF = load_dataset()
ANA = DF[DF["title"] == CLIP_TITLE].iloc[0].to_dict()  # yt_c0143, 198 seconds


def decision_json(end: float = 55) -> str:
    return json.dumps(
        [
            {
                "clip_id": ANA["clip_id"],
                "action": "age-restrict",
                "policy_reason": "Graphic violence shown in a news or documentary context",
                "reason_code": "VC-03",
                "evidence": [{"start": 16, "end": end, "description": "A bottle is thrown at officers."}],
            }
        ]
    )


VALID_REPLY = decision_json()
PAST_THE_CLIP_REPLY = decision_json(end=250)  # 250s on a 198s clip


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the model call with a queue of canned replies, and record every call."""

    class FakeModel:
        def __init__(self):
            self.replies: list[str] = []
            self.calls: list[list[dict]] = []

        def __call__(self, messages, model_id, api_key, **kwargs):
            self.calls.append(messages)
            if not self.replies:
                raise AssertionError("the model was called more times than replies were queued")
            return {"role": "assistant", "content": self.replies.pop(0)}

    fake = FakeModel()
    monkeypatch.setattr(tb, "_call_openrouter", fake)
    monkeypatch.setattr(tb, "load_settings", lambda: ("test-key", "test/model"))
    return fake


def test_valid_first_reply_is_not_retried(fake_model):
    fake_model.replies = [VALID_REPLY]

    decisions = tb.triage_batch([ANA])

    assert len(fake_model.calls) == 1
    assert decisions[ANA["clip_id"]].action == "age-restrict"


def test_broken_reply_is_retried_once_with_the_error(fake_model):
    fake_model.replies = [ANA_RAW_REPLY, VALID_REPLY]

    decisions = tb.triage_batch([ANA])

    assert len(fake_model.calls) == 2
    assert decisions[ANA["clip_id"]].action == "age-restrict"

    # The retry carries the original prompt, the model's own broken reply, and the error.
    first_call, retry_call = fake_model.calls
    assert [m["role"] for m in retry_call] == ["user", "assistant", "user"]
    assert retry_call[0] == first_call[0]
    assert retry_call[1]["content"] == ANA_RAW_REPLY
    assert f"{ANA['clip_id']}: reply is not valid JSON" in retry_call[2]["content"]


def test_schema_error_is_sent_back_on_the_retry(fake_model):
    fake_model.replies = [PAST_THE_CLIP_REPLY, VALID_REPLY]

    decisions = tb.triage_batch([ANA])

    assert len(fake_model.calls) == 2
    assert decisions[ANA["clip_id"]].evidence[0].end == 55
    retry_message = fake_model.calls[1][2]["content"]
    assert "runs past the clip's end (198.0s)" in retry_message


def test_second_broken_reply_raises_and_never_defaults(fake_model):
    fake_model.replies = [ANA_RAW_REPLY, PAST_THE_CLIP_REPLY, VALID_REPLY]

    with pytest.raises(tb.TriageBatchError) as caught:
        tb.triage_batch([ANA])

    # Exactly two calls: the third (valid) reply was never asked for.
    assert len(fake_model.calls) == 2
    assert fake_model.replies == [VALID_REPLY]

    error = caught.value
    assert error.decisions == {}  # no label was invented for the clip
    assert "runs past the clip's end (198.0s)" in error.errors[ANA["clip_id"]]
    assert "not valid JSON" in error.first_attempt_errors[ANA["clip_id"]]
    assert "after 2 attempts" in str(error)
