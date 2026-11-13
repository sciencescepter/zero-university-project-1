import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "score": [88, 92, 79],
        }
    )
    print(df)
    print(f"\nAverage score: {df['score'].mean():.1f}")


if __name__ == "__main__":
    main()
