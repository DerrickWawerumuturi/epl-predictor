# Step 1: Imports

import pandas as pd
from pathlib import Path
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


# Step 2: Data Preparation

current_dir = Path.cwd()
parent_dir = current_dir.parent

sub_folder = os.path.join(parent_dir, "data")
tgt_subfolder = os.path.join(sub_folder, "tournaments")
tgt_parent_subfolder = os.path.join(tgt_subfolder, "outputs_euro")
target = os.path.join(tgt_parent_subfolder, "processed", 'player_agg_euro_2024.parquet')

DATA = target

if os.path.exists(DATA):
    print(f"The path '{DATA}' exists.")
else:
    print(f"The path '{DATA}' does not exist.")

# Load the data from the outputs_euro

df = pd.read_parquet(DATA)

df = df[df['minutes'].fillna(0) > 0].copy()

# get the roles for the players
def to_role(p):
    if isinstance(p, str):
        p = p.split(",")[0].strip()
    if p in ("GK",):                     return "GK"
    if p in ("DF","FB","WB","CB"):       return "DF"
    if p in ("MF","DM","CM","AM","WM"):  return "MF"
    if p in ("FW","CF","WF","SS"):       return "FW"
    return 'UNK'

df['role'] = df['primary_pos'].apply(to_role)
df = df[df["role"] != "UNK"].copy()

# Helpers
def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=s.index, dtype="Float64")
    out = (s - mu) / sd
    return out.astype("Float64")

# Ensure role exists
if "role" not in df.columns:
    pos = df["primary_pos"].fillna("")
    df["role"] = (
        np.select(
            [
                pos.str.startswith(("FW","ST","CF","LW","RW")),
                pos.str.startswith(("MF","CM","DM","AM","LM","RM")),
                pos.str.startswith(("DF","CB","RB","LB","RWB","LWB")),
                pos.str.startswith(("GK",)),
            ],
            ["FW","MID","DEF","GK"],
            default="MID",
        )
        .astype("string")
    )

# Making sure source columns are numeric
for c in ["gls_90","xg_90","starter_rate"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Build targets with explicit nullable dtypes
is_fw = df["role"].eq("FW").fillna(False)

# target_cls: Int8 (nullable)
df["target_cls"] = pd.Series(pd.NA, index=df.index, dtype="Int8")
# Forwards: goals per 90 > xG per 90
comp_fw = (df.loc[is_fw, "gls_90"] > df.loc[is_fw, "xg_90"]).astype("Int8")
df.loc[is_fw, "target_cls"] = comp_fw

# Non-forwards: starter_rate >= median within non-FW
non_fw = ~is_fw
thr = df.loc[non_fw, "starter_rate"].median(skipna=True)
df.loc[non_fw, "target_cls"] = (df.loc[non_fw, "starter_rate"] >= thr).astype("Int8")

# target_reg: Float64 (nullable)
df["target_reg"] = pd.Series(pd.NA, index=df.index, dtype="Float64")

# Forwards: z-score of (gls_90 - xg_90)
df["delta_fw"] = (df["gls_90"] - df["xg_90"]).astype("Float64")
df.loc[is_fw, "target_reg"] = zscore(df.loc[is_fw, "delta_fw"]).astype("Float64")

# Others: z-score of starter_rate within each role (DEF/MID/GK separately)
df["delta_other"] = df["starter_rate"].astype("Float64")
df.loc[non_fw, "target_reg"] = (
    df.loc[non_fw]
      .groupby("role", group_keys=False)["delta_other"]
      .apply(zscore)
      .astype("Float64")
)

# Final modeling frame
df_model = (
    df.replace([np.inf, -np.inf], np.nan)
      .dropna(subset=["target_cls", "target_reg"])
      .copy()
)
# Features:

num_feats = [
    "age","minutes","nineties","mp","starts",
    "minutes_share","starter_rate",
    "xg_90","xag_90","npxg_90","npxg_xag_90",
    # light totals (scaled by 90s anyway, but fine to include)
    "gls_90","ast_90","ga_90",
]

cat_feats = ['role']
# Hot encode the features
df_model = df_model.replace([np.inf, -np.inf], np.nan).copy()
y = df_model["target_reg"]
mask = y.notna()
y = y.loc[mask]
X = df_model.loc[mask, num_feats + cat_feats]

# the target variables for training
y = df_model["target_reg"].astype(float)


# Step 3: Train/test split md

# train split
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=df_model.loc[X.index, "role"]
)

# validation split
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=df_model.loc[X_temp.index, "role"]
)


# Step 4: Model definition and preprocessing md
# 	1.	Use Ridge to find stable feature importances.
# 	2.	Use Random Forest to rank players and make performance predictions.
# 	3.	Compare both — if they agree, your model’s probably capturing real signal.
# keep only numeric columns that have at least one observed value
valid_num = [c for c in num_feats if X[c].notna().any()]
dropped = sorted(set(num_feats) - set(valid_num))
if dropped:
    print("Dropping all-NaN numeric features:", dropped)

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

pre = ColumnTransformer([
    ("num", num_pipe, valid_num),
    ("cat", cat_pipe, cat_feats),
])


models = {
    "linreg": LinearRegression(),
    "ridge": RidgeCV(alphas=np.logspace(-4, 4, 25), cv=5),
    "rf": RandomForestRegressor(
        n_estimators=600, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
}

pipes =  { name: Pipeline([("pre", pre), ("model", m)]) for name, m in models.items() }


# Step 5: Train and validate the model
def eval_reg(y_true, y_pred, label=''):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label:>8} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}

scores = {}
for name, pipe in pipes.items():
    # train the model
    pipe.fit(X_train, y_train)
    pred_val = pipe.predict(X_val)
    scores[name] = eval_reg(y_val, pred_val)

best_name = min(scores, key=lambda k: scores[k]["rmse"])
best_pipe = pipes[best_name]
print(f"\nBest on VAL: {best_name} → {scores[best_name]}")


# # Step 6: Test the model
pred_test = best_pipe.predict(X_test)
eval_reg(y_test, pred_test, "TEST")
importances = best_pipe.named_steps["model"].feature_importances_
feature_names = pre.get_feature_names_out()
sorted_idx = importances.argsort()

plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx][-15:], importances[sorted_idx][-15:])
plt.xlabel("Feature Importance")
plt.title("Top 15 Most Important Features")
plt.show()

# Step 7: Rank all euro players by the predicted performance
df_model["predicted_performance"] = best_pipe.predict(X)

df_ranked = df_model.sort_values(by="predicted_performance", ascending=False)

# Reset index for neatness
df_ranked = df_ranked.reset_index(drop=True)

# Display top 15
top_n = 15
top_players = df_ranked.head(top_n)

plt.figure(figsize=(10,6))
plt.barh(top_players["player_name"][::-1], top_players["predicted_performance"][::-1])
plt.xlabel("Predicted Performance")
plt.title(f"Top {top_n} Players by Expected Performance")
plt.show()
df_model["league"].unique()