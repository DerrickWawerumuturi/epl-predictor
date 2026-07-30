# XG-PREDICTOR — Frontend

Dark, editorial single-page UI for the [epl-predictor](https://github.com/DerrickWawerumuturi/epl-predictor) model. Displays the 87 Euro-2024 players ranked for the 2025–26 Premier League season with role filters, search, sorting, a top-3 podium, and per-role rating distributions.

## Run

```bash
npm install
npm run dev       # local dev at http://localhost:5173
npm run build     # production build → dist/
```

A pre-built `dist/` is included — open `dist/index.html` directly in a browser for a zero-setup preview.

## Data

`src/data/players.js` is generated from the model output `backend/models/premier_league_player_ratings_5pt.csv`. Re-run the model, regenerate the JSON, rebuild.

## Stack

React 18 + Vite 5, no UI framework — hand-rolled CSS design system (Space Grotesk / JetBrains Mono, grid backdrop, pixel-motif accents).
