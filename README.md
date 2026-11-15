# zero-university-project-1

First project for the Zero University Applied AI Engineer program: a small Python project using pandas.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your real values
```

## Run

```bash
python main.py
```

## Verdict

Clip identity (assumed): yt_c0143, "Clashes outside parliament as fuel price protest turns violent", is a best match for Ana's protest clip, not a confirmed one. It is the only title in the dataset containing "protest", but nothing in the data names Ana.

Raw model reply: [outputs/ana_protest_raw_reply.txt](outputs/ana_protest_raw_reply.txt)

Label: No, the model did not remove the clip; it labelled it age-restrict, which matches the ground-truth label.

Timestamps: Yes, 00:30 and 01:50 are both inside the clip's 198-second duration, and both overlap the ground-truth violent segment (0:20-2:20).

## Decision schema

The model's free-text reply breaks the enforcement policy in several ways: it has no policy reason, its evidence moments are single points with no end, and nothing stops a made-up label. Full list: [outputs/enforcement_gaps.md](outputs/enforcement_gaps.md).

[src/decision_schema.py](src/decision_schema.py) defines a decision with:

- **action**: only `keep`, `age-restrict` or `remove`
- **policy_reason**: one of the seven approved wordings, exactly as written, and only with the actions that reason permits
- **evidence**: timestamped moments (start and end in seconds, plus a description); `age-restrict` and `remove` must cite at least one

`validate_decision(data, clip_duration_seconds)` takes the clip's runtime from the dataset, never from the model. It rejects moments that run backwards, have zero length, or run past the end of the clip.

### Tests

```bash
.venv/bin/python -m pytest -v
```

Four tests, using Ana's real reply: [outputs/pytest_output.txt](outputs/pytest_output.txt) (4 passed).
