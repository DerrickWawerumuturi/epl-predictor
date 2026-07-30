# XG-PREDICTOR — Frontend

Single-page UI for the [epl-predictor](https://github.com/DerrickWawerumuturi/epl-predictor) Euro Impact Index. Shows all 256 Euro 2024 players with ≥180 tournament minutes, ranked **within their own position**, with role filters, search, sorting, a top-3 podium and per-role rating distributions.

The rating is a percentile within position mapped to 1.0–5.0 — not a prediction. See [MODEL_CARD.md](../MODEL_CARD.md) for what the index measures and what it does not.

## Run

```bash
npm install
npm run dev       # http://localhost:5173
npm run build     # production build → dist/ (not committed)
```

## Data

`src/data/players.js` is generated from `backend/models/outputs/euro_impact_index.csv`. To refresh it:

```bash
cd ../backend/models && python index_score.py
```

then regenerate the JS module from the resulting CSV.

## Stack

React 18 + Vite 5, no UI framework — hand-rolled CSS design system (Space Grotesk / JetBrains Mono, grid backdrop, pixel-motif accents).
