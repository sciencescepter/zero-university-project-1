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
