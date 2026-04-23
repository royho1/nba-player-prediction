"""
Data Cleaning Pipeline for NBA Player Performance Prediction

Loads raw NBA player data, filters for players with 5 consecutive post-draft
seasons, consolidates per-season rows into a single row per player, and outputs
a cleaned CSV ready for modeling.
"""

import pandas as pd
import os


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load raw NBA data and filter to players who played 5 consecutive
    seasons after being drafted.

    Parameters
    ----------
    filepath : str
        Path to the raw all_seasons.csv file.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only qualifying players and seasons 1-5.
    """
    data = pd.read_csv(filepath)

    # Extract start year from season string (e.g., "2015-16" -> 2015)
    data["season"] = data["season"].str[:-3]

    # Remove undrafted players
    data = data[~data.isin(["Undrafted"]).any(axis=1)]

    # Convert to numeric
    data["season"] = pd.to_numeric(data["season"])
    data["draft_year"] = pd.to_numeric(data["draft_year"])

    # Identify players who played their 5th consecutive season
    playerlist = []
    for i in range(len(data)):
        if data["season"].iloc[i] - data["draft_year"].iloc[i] == 4:
            playerlist.append(data["player_name"].iloc[i])

    # Keep only those players and only their first 5 seasons
    name_mask = data["player_name"].isin(playerlist)
    diff_mask = (data["season"] - data["draft_year"]) <= 4
    train = data[name_mask & diff_mask].sort_values(by="player_name")

    # Update legacy team abbreviations
    team_map = {
        "NJN": "BKN",
        "SEA": "OKC",
        "VAN": "MEM",
        "CHO": "CHA",
        "CHH": "CHA",
        "NOH": "NOP",
        "NOK": "NOP",
    }
    train["team_abbreviation"] = train["team_abbreviation"].replace(team_map)

    # Keep only players with exactly 5 seasons of data
    counts = train["player_name"].value_counts()
    counts5 = counts[counts == 5].index
    train5 = train[train["player_name"].isin(counts5)]

    return train5


def build_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot per-season rows into a single row per player with season-specific
    point columns and rookie-year features.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered DataFrame from load_and_clean().

    Returns
    -------
    pd.DataFrame
        One row per player with columns: age, team_abbreviation, draft_round,
        draft_number, player_height, player_weight, ptsseason1-5.
    """
    season_counts = (
        df.groupby("player_name")["season"].nunique().sort_values(ascending=False)
    )
    players_with_5 = season_counts[season_counts >= 5].index
    df_5 = df[df["player_name"].isin(players_with_5)]

    df_sorted = df_5.sort_values(by=["player_name", "season"])
    df_sorted["season_index"] = df_sorted.groupby("player_name").cumcount() + 1

    # Pivot points per game into separate columns
    pts_pivot = df_sorted.pivot(
        index="player_name", columns="season_index", values="pts"
    )
    pts_pivot.columns = [f"ptsseason{i}" for i in pts_pivot.columns]

    # Use rookie season features
    first_season = df_sorted[df_sorted["season_index"] == 1].set_index(
        "player_name"
    )
    features = first_season[
        [
            "age",
            "team_abbreviation",
            "draft_round",
            "draft_number",
            "player_height",
            "player_weight",
        ]
    ]

    model_data = pd.concat([features, pts_pivot], axis=1).dropna()
    return model_data


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    raw_path = os.path.join(data_dir, "all_seasons.csv")

    if not os.path.exists(raw_path):
        print(
            f"Error: {raw_path} not found. Download the dataset from Kaggle "
            "and place all_seasons.csv in the data/ directory."
        )
        return

    print("Loading and cleaning data...")
    train5 = load_and_clean(raw_path)
    print(f"  Players with 5 consecutive seasons: {train5['player_name'].nunique()}")

    print("Building model-ready dataset...")
    model_data = build_model_data(train5)
    print(f"  Final dataset shape: {model_data.shape}")

    out_path = os.path.join(data_dir, "train5final.csv")
    model_data.to_csv(out_path, index=True)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
