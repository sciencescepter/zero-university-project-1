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

Raw model reply: [outputs/ana_protest_raw_reply.txt](outputs/ana_protest_raw_reply.txt)

Label: No, the model did not remove the clip; it labelled it age-restrict, which matches the ground-truth label.

Timestamps: Yes, 00:30 and 01:50 are both inside the clip's 198-second duration, and both overlap the ground-truth violent segment (0:20-2:20).
