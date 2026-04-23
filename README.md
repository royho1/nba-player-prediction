# NBA Player Performance Prediction

Predicting NBA players' fifth-season scoring output using their first four seasons of performance data, and classifying whether a player's fifth season will be their statistical peak.

**Course:** STA 141C - Advanced Statistical Computing, UC Davis  
**Team:** Ben Grigsby, Elliot Weisberg, Evan Carlson, Nolan Glavis, Roy Ho  
**Date:** June 2025

---

## Overview

The NBA generates over $11 billion in annual revenue, and player evaluation drives decisions worth hundreds of millions of dollars in trades, contracts, and draft picks. This project investigates two questions:

1. **Regression:** Can we predict a player's fifth-season points per game from their first four seasons, physical attributes, and draft position?
2. **Classification:** Can we predict whether a player's fifth season will be their best scoring season out of the five?

We used a [Kaggle dataset](https://www.kaggle.com/datasets/justinas/nba-players-data) containing 12,843 player-season records from 1996 to 2022. After filtering for players who played five consecutive seasons post-draft, our cleaned dataset contained 601 players.

## Key Results

### Regression

| Model | RMSE | R² |
|---|---|---|
| OLS (Linear Additive) | **3.001** | **0.767** |
| Simple Decision Tree | 5.13 | 0.403 |
| Random Forest | 3.657 | 0.696 |
| Random Forest (Tuned) | 3.538 | 0.715 |
| Gradient Boosted Trees | 3.402 | 0.737 |
| Gradient Boosted Trees (Tuned) | 3.407 | 0.736 |

The OLS model outperformed all tree-based models, indicating the relationship between early-career scoring and fifth-season output is primarily linear. Season 4 points per game was the strongest predictor by a wide margin.

### Classification

| Model | Accuracy |
|---|---|
| LDA | 0.626 |
| QDA | 0.622 |
| Random Forest | 0.734 |
| **Random Forest (Tuned)** | **0.735** |
| Random Forest w/ Feature Engineering | 0.646 |
| XGBoost | 0.647 |
| XGBoost (Tuned) | 0.683 |

The tuned Random Forest classifier achieved the best accuracy at 73.5%, though all models struggled with the imbalanced class distribution (roughly 2:1 ratio of "not best" to "best" season).

## Methods

### Data Cleaning
- Filtered to players who played 5 consecutive seasons after being drafted
- Consolidated per-season rows into a single row per player with season-specific point columns
- Updated legacy team abbreviations (e.g., NJN to BKN, SEA to OKC)
- Dropped team as a feature to prevent overfitting across 30 categories

### Exploratory Data Analysis
- Correlation heatmap analysis of features vs. target
- K-Means clustering (k=3) to identify natural player groupings: high-performance, medium-performance, and low-performance tiers
- Elbow plot for optimal cluster selection

### Regression Models
- Ordinary Least Squares (baseline)
- Decision Tree, Random Forest, Gradient Boosted Trees
- Hyperparameter tuning via GridSearchCV
- 3-fold cross-validation for model comparison

### Classification Models
- LDA and QDA (baselines)
- Random Forest with RandomizedSearchCV tuning
- XGBoost with class imbalance handling (`scale_pos_weight`)
- Feature engineering: height*weight interaction, improving player indicator

## Project Structure

```
nba-player-prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── train5final.csv        # Cleaned dataset (601 players)
├── notebooks/
│   └── nba_analysis.ipynb     # Full analysis notebook
├── src/
│   ├── data_cleaning.py       # Data preprocessing pipeline
│   ├── regression.py          # Regression models (OLS, trees, ensembles)
│   └── classification.py     # Classification models (LDA, QDA, RF, XGBoost)
└── figures/
    └── README.md              # Generated figures from the analysis
```

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Run the full notebook
jupyter notebook notebooks/nba_analysis.ipynb

# Or run individual scripts
python src/regression.py
python src/classification.py
```

## Tech Stack
- Python 3.9+
- pandas, NumPy, scikit-learn, XGBoost, statsmodels
- matplotlib, seaborn
- scipy

## Limitations
- Dataset reduced from 12,843 to 601 players after filtering for 5 consecutive seasons
- Undrafted players (~25% of the NBA) are excluded entirely
- Points per game does not account for minutes played, which heavily influences scoring volume
- Classification target is imbalanced (roughly 2:1), which affected all classifier performance

## References

Citrautas, Justin. *NBA Players.* Kaggle, 2023. [https://www.kaggle.com/datasets/justinas/nba-players-data](https://www.kaggle.com/datasets/justinas/nba-players-data)
