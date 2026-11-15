"""Decision schema for the Violent Content enforcement policy.

A decision has exactly one action, one approved policy reason, and a list of
timestamped evidence moments. The rules mirror
"Violent Content Enforcement Actions and Policy Reasons.md".

The clip's runtime is never taken from the decision. It is passed in by us,
from the dataset's duration_seconds (the upload record), through
validate_decision / validate_decision_json.

Every rejection is raised by the schema itself, as a pydantic ValidationError.
"""

import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    model_validator,
)

Action = Literal["keep", "age-restrict", "remove"]

# Approved policy reasons: wording -> (code, actions that wording permits).
# The wording must match exactly. Paraphrases and bare codes are rejected.
APPROVED_REASONS: dict[str, tuple[str, set[str]]] = {
    "No violent content requiring enforcement": ("VC-00", {"keep"}),
    "Non-graphic or incidental violence": ("VC-01", {"keep"}),
    "Graphic violence with educational or artistic context": ("VC-02", {"age-restrict"}),
    "Graphic violence shown in a news or documentary context": ("VC-03", {"age-restrict"}),
    "Graphic violence without context": ("VC-04", {"remove"}),
    "Violence posted to glorify or celebrate harm": ("VC-05", {"remove"}),
    "Incitement or credible threat of violence": ("VC-06", {"remove"}),
}

_DURATION_KEY = "clip_duration_seconds"


class EvidenceMoment(BaseModel):
    """One cited moment: a span of the clip in seconds, plus what happens in it."""

    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0, description="Seconds from the start of the clip.")
    end: float = Field(ge=0, description="Seconds from the start of the clip.")
    description: str = Field(min_length=1, description="One plain sentence of what is shown or said.")

    @model_validator(mode="after")
    def start_before_end(self):
        if self.start >= self.end:
            raise ValueError(
                f"moment must start before it ends (start={self.start}, end={self.end})"
            )
        return self

    @model_validator(mode="after")
    def within_clip(self, info: ValidationInfo):
        duration = (info.context or {}).get(_DURATION_KEY)
        if duration is None:
            raise ValueError(
                "clip runtime was not supplied; validate through validate_decision"
            )
        if self.end > duration:
            raise ValueError(
                f"moment {self.start}-{self.end}s runs past the clip's end ({duration}s)"
            )
        return self


class Decision(BaseModel):
    """A moderation decision. Extra fields (such as a model-supplied runtime) are refused."""

    model_config = ConfigDict(extra="forbid")

    action: Action
    policy_reason: str = Field(description="Approved reason wording, exactly as written in the policy.")
    reason_code: str | None = Field(default=None, description="Optional code; must match the wording if given.")
    evidence: list[EvidenceMoment] = Field(default_factory=list)

    @model_validator(mode="after")
    def follows_policy(self):
        if self.policy_reason not in APPROVED_REASONS:
            raise ValueError(f"policy_reason is not an approved wording: {self.policy_reason!r}")

        code, permitted_actions = APPROVED_REASONS[self.policy_reason]
        if self.reason_code is not None and self.reason_code != code:
            raise ValueError(f"reason_code {self.reason_code!r} does not match wording (expected {code})")
        if self.action not in permitted_actions:
            raise ValueError(f"{code} does not permit action {self.action!r}")

        if self.action in ("age-restrict", "remove") and not self.evidence:
            raise ValueError(f"an {self.action!r} decision must cite at least one moment")
        return self


def _context(clip_duration_seconds: float) -> dict:
    return {_DURATION_KEY: float(clip_duration_seconds)}


def validate_decision(data: dict, clip_duration_seconds: float) -> Decision:
    """Validate a decision (as a dict) against the schema and the clip's real runtime.

    clip_duration_seconds must come from the upload record (the dataset),
    never from the model's reply. Raises pydantic.ValidationError on rejection.
    """
    return Decision.model_validate(data, context=_context(clip_duration_seconds))


def validate_decision_json(json_text: str, clip_duration_seconds: float) -> Decision:
    """Same as validate_decision, for a decision stored as a JSON string."""
    return Decision.model_validate_json(json_text, context=_context(clip_duration_seconds))


def _mmss_to_seconds(stamp: str) -> float:
    minutes, seconds = stamp.split(":")
    return int(minutes) * 60 + float(seconds)


def parse_raw_reply(text: str) -> dict:
    """Turn the model's free-text reply into a dict the schema can check.

    This does no fixing: a missing reason stays missing, and a single
    timestamp like [00:30] becomes a moment whose start equals its end.
    """
    data: dict = {}

    label = re.search(r"^Label:\s*(\S+)", text, re.MULTILINE)
    if label:
        data["action"] = label.group(1)

    reason = re.search(r"^(?:Policy reason|Reason):\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if reason:
        data["policy_reason"] = reason.group(1).strip()

    evidence = []
    for match in re.finditer(r"^-\s*\[(\d+:\d+)(?:-(\d+:\d+))?\]\s*(.+)$", text, re.MULTILINE):
        start = _mmss_to_seconds(match.group(1))
        end = _mmss_to_seconds(match.group(2)) if match.group(2) else start
        evidence.append({"start": start, "end": end, "description": match.group(3).strip()})
    data["evidence"] = evidence

    return data


__all__ = [
    "APPROVED_REASONS",
    "Decision",
    "EvidenceMoment",
    "ValidationError",
    "parse_raw_reply",
    "validate_decision",
    "validate_decision_json",
]
