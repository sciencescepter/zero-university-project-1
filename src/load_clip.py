"""Load the YouTube triage dataset and pick out the protest clip by its full title."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "youtube_triage_labelled_test_set_200_clips.csv"

# Full title of the clip to select. This is the only title in the dataset that
# contains "protest". It has not been confirmed as Ana's clip, so change it here
# if the course names a different title.
CLIP_TITLE = "Clashes outside parliament as fuel price protest turns violent"


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def select_clip(df: pd.DataFrame, title: str = CLIP_TITLE) -> pd.Series:
    """Return the one row whose title matches exactly, or fail loudly."""
    matches = df[df["title"] == title]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly 1 row titled {title!r}, found {len(matches)}.")
    return matches.iloc[0]


if __name__ == "__main__":
    df = load_dataset()
    print(f"Rows: {len(df)}")

    print("\nTitles containing 'protest':")
    protest_titles = df.loc[df["title"].str.contains("protest", case=False, na=False), "title"]
    for title in protest_titles:
        print(f"- {title}")

    clip = select_clip(df)
    transcript = clip["transcript_timestamped"]
    print(f"\nSelected clip: {clip['clip_id']} ({clip['duration_seconds']:.0f} seconds)")
    print(f"transcript_timestamped is a {type(transcript).__name__}, {len(transcript)} characters:\n")
    print(transcript)
