"""Show which clip call_model.py sends, and prove the ground-truth columns stay out.

Builds the exact prompt call_model.py sends (no model call) and checks it.
"""

from call_model import build_prompt
from load_clip import load_dataset, select_clip

GROUND_TRUTH_COLUMNS = [
    "ground_truth_label",             # J
    "ground_truth_violent_segments",  # K
    "ground_truth_policy_reason",     # L
]

clip = select_clip(load_dataset())
prompt = build_prompt(clip)

# The label options (keep / age-restrict / remove) are part of the instructions,
# so only check the clip data that follows them.
clip_data = prompt[prompt.index("Clip duration:"):]

print(f"Clip sent to the model: {clip['clip_id']}")
print(f"Title: {clip['title']}\n")

print("Ground-truth columns in the prompt?")
for column in GROUND_TRUTH_COLUMNS:
    value = str(clip[column])
    status = "FOUND - LEAK" if value in clip_data else "not sent"
    print(f"- {column}: {status}")

print("\nEvidence timestamps in this clip's frame captions:")
captions = clip["frame_captions_timestamped"]
for timestamp in ["[00:30]", "[01:50]"]:
    caption = next((c.strip() for c in captions.split("|") if c.strip().startswith(timestamp)), None)
    print(f"- {timestamp}: {caption[len(timestamp):].strip() if caption else 'NOT IN THIS CLIP'}")
