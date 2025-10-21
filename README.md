# ⚽ EPL Player Performance Predictor

## 📘 Overview
This project builds a **machine learning model** that predicts and ranks football players’ expected performances on the 2025-2026 premier league based on their euros performance.  
It uses **scikit-learn regression models** (Linear Regression, Ridge Regression, and Random Forest Regressor) trained on per-90 metrics, then converts the model’s output into a **1–5 rating scale** to make the results interpretable.

The final output is a CSV file containing **Premier League players** ranked from highest to lowest expected performance, along with their **age, team, and role**.

---

## 🎯 Objective
Predict how well a player is expected to perform (relative to others) based on their playing time, efficiency, and attacking metrics — and assign each a human-readable rating between **1.0 and 5.0**.

---

## 🧩 Key Steps
1. **Data Preparation**
   - Load player statistics into a Pandas DataFrame (`df_model`).
   - Include numerical and categorical features.
   - Define the target variable (`target_reg`) — a standardized performance measure.

2. **Feature Encoding**
   - Numeric features are kept as floats.
   - The categorical `role` column is one-hot encoded using `pd.get_dummies()`.

3. **Train–Validation–Test Split**
   - Data is split into train, validation, and test sets using `train_test_split`.
   - Stratification by player role ensures balanced representation across positions.

4. **Model Training**
   Three models are trained and evaluated:
   - **Linear Regression** — captures simple linear relationships.
   - **Ridge Regression (RidgeCV)** — adds regularization to prevent overfitting.
   - **Random Forest Regressor** — an ensemble of decision trees that models non-linear relationships.

5. **Evaluation Metrics**
   - **MAE (Mean Absolute Error)** — measures average prediction error.
   - **RMSE (Root Mean Square Error)** — penalizes large errors.
   - **R² (R-squared)** — measures variance explained by the model.

➜ **Random Forest** performs best.

6. **Performance Scaling**
- The model’s predicted scores are scaled into a 1.0–5.0 range using the 5th and 95th percentiles.
- This creates a clean, human-friendly rating scale:
  - 1.0 = low performance  
  - 3.0 = average  
  - 5.0 = top performance  

7. **Premier League Filtering**
- FBref player stats are scraped via `pd.read_html()` from  
  [https://fbref.com/en/comps/9/stats/Premier-League-Stats](https://fbref.com/en/comps/9/stats/Premier-League-Stats)
- Player names are cleaned with `clean_name()` and merged with the prediction DataFrame to keep only EPL players.
- Age and team are extracted and added as new columns.

8. **Final Output**
The final CSV (`premier_league_player_ratings_final.csv`) includes:
- `player_name`
- `age`
- `team`
- `role`
- `predicted_performance`
- `rating_5pt`

---
## Running the file
` python epl_player_predictor.py `

## 🧠 How Random Forest Works (Simplified)
- Trains many decision trees on random subsets of data.
- Each tree predicts a performance value.
- The final prediction is the **average of all trees**.
- This reduces overfitting and captures non-linear patterns in player metrics.

---

## ⚙️ Tech Stack
- **Python 3.11**
- **Pandas** — data wrangling
- **NumPy** — numerical operations
- **Scikit-learn** — modeling & evaluation
- **Requests / lxml** — web scraping (FBref)
- **Matplotlib (optional)** — visualization

---

## 📂 Output Example

| Player Name | Age | Team | Role | Predicted Performance | Rating (1–5) |
|--------------|-----|------|------|-----------------------|---------------|
| Erling Haaland | 24 | Man City | FW | 0.842 | 4.9 |
| Bukayo Saka | 23 | Arsenal | FW | 0.715 | 4.6 |
| Rodri | 28 | Man City | MF | 0.522 | 4.2 |
| James Maddison | 27 | Tottenham | MF | 0.314 | 3.8 |
| Declan Rice | 25 | Arsenal | MF | 0.181 | 3.5 |

---

## 💾 Running the Script
1. Install dependencies:
```bash
pip install pandas numpy scikit-learn lxml requests
