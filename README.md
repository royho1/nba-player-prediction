# Data

## Included

- `train5final.csv` - Cleaned dataset of 601 NBA players who played 5 consecutive seasons post-draft. Each row contains one player with their rookie-year attributes (age, team, draft position, height, weight) and points per game for seasons 1 through 5. This is the input used by all regression and classification models.

## Optional

To re-run the data cleaning pipeline from scratch, download `all_seasons.csv` from [Kaggle - NBA Players](https://www.kaggle.com/datasets/justinas/nba-players-data) and place it in this directory, then run:

```bash
python src/data_cleaning.py
```

The raw dataset contains 12,843 player-season records from the 1996 to 2022 NBA seasons.
