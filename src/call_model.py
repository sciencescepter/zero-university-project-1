"""Send one message to the model named in config.yaml, via OpenRouter.

The API key comes from .env (OPENROUTER_API_KEY) and the model from
config.yaml (model_id). Neither is written in this file.
"""

import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from load_clip import load_dataset, select_clip

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Cap the reply length. Without it, OpenRouter reserves credit for the model's
# full maximum (131k tokens) and rejects the call if the key can't cover that.
MAX_REPLY_TOKENS = 2000


def load_settings() -> tuple[str, str]:
    """Return (api_key, model_id), failing with a clear message if either is missing."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is missing. Add it to .env.")

    with open(PROJECT_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f) or {}
    model_id = config.get("model_id")
    if not model_id:
        sys.exit("model_id is missing. Add it to config.yaml.")

    return api_key, model_id


def call_model(message: str) -> str:
    api_key, model_id = load_settings()
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": message}],
        max_tokens=MAX_REPLY_TOKENS,
    )
    return response.choices[0].message.content


def build_prompt(clip) -> str:
    """Build the moderation prompt from only columns G, H and I.

    The ground-truth columns (J, K, L) are deliberately never read here, so the
    model can't see the answers. Title, channel and category are left out too.
    """
    duration = clip["duration_seconds"]               # column G
    transcript = clip["transcript_timestamped"]       # column H
    frame_captions = clip["frame_captions_timestamped"]  # column I

    return f"""You are reviewing a YouTube video clip for a content moderation team.

Choose exactly one label for the clip:
- keep: fine for all audiences
- age-restrict: allowed, but only for adult viewers
- remove: violates policy and should be taken down

Then list the timestamps of the specific moments you used as evidence for your label.

Clip duration: {duration} seconds

Transcript (timestamped):
{transcript}

Frame captions (timestamped):
{frame_captions}"""


if __name__ == "__main__":
    clip = select_clip(load_dataset())
    print(call_model(build_prompt(clip)))
