# F1 2026 Prediction System

This project is an end-to-end Formula 1 race and championship prediction system for the 2026 season.

It models qualifying performance, simulates race outcomes, and projects full season standings using probabilistic methods.

The system was built by designing modular Python components, integrating multiple datasets, and running the full pipeline in a notebook environment for analysis and validation.

---

## Overview

The model predicts:

* qualifying results for every race
* race finishing order for every round
* win, podium, top 5, and points probabilities
* expected finishing positions
* full driver championship standings
* full constructor championship standings

The approach combines:

* historical race data
* 2026 session and testing data
* custom priors for drivers, teams, engines, tracks, upgrades, and reliability

---

## How the system works

### 1. Data loading

Handled in `data_loader.py`.

This step loads and standardizes:

* historical datasets
* 2026 lap and session data
* prior assumptions for drivers, teams, engines, tracks, and reliability

---

### 2. Feature engineering

Handled in `feature_engineering.py`.

Key steps:

* construct driver and team performance signals
* extract pace metrics from 2026 session data
* merge priors with real performance data
* generate a unified dataset for modeling

Because of the 2026 regulation reset, current-season signals are weighted more heavily than historical team performance.

---

### 3. Qualifying model

Implemented in `quali_model.py`.

The model predicts:

* relative driver pace
* qualifying positions for each race

Inputs include:

* 2026 pace signals
* driver skill priors
* team and engine strength

Output:

* predicted qualifying grid for each round

---

### 4. Race simulation

Implemented in `race_simulation.py`.

Each race is simulated using Monte Carlo methods.

The simulation includes:

* qualifying grid influence
* driver performance variability
* team and engine performance
* reliability and DNF probability
* randomness from race conditions

Each race is run many times to produce:

* expected finishing position
* win probability
* podium probability
* top 5 probability
* points probability
* DNF probability

---

### 5. Season simulation

The full season is built by simulating each race and aggregating the results.

This produces:

* total expected points
* race-by-race projections
* consistent season outcomes

---

### 6. Standings calculation

Handled in `standings.py`.

The system computes:

* driver championship standings
* constructor championship standings

Based on:

* expected points from all simulated races

---

## Main outputs

The system produces:

* qualifying predictions for every round
* race summaries for every round
* full probability tables for each race
* combined datasets across the season
* projected driver standings
* projected constructor standings
* expected points by race

---

## Required data

The project expects the following datasets.

### Historical data

A cleaned master dataset such as:

* `model_ready_combined_2009_2025.csv`

---

### Priors (2026 assumptions)

* `driver_priors_2026.csv`
* `team_priors_2026.csv`
* `engine_priors_2026.csv`
* `track_priors_2026.csv`
* `reliability_priors_2026.csv`
* `upgrade_priors_2026.csv`
* `team_engine_map_2026.csv`
* `driver_team_map_2026.csv`

---

### 2026 data

* pre-season testing data
* race weekend session data
* lap times
* qualifying results
* weather data

---

## Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the project

Run the full system:

```
python main.py
```

Run a single race:

```
python main.py --race 3
```

Run fewer simulations for faster testing:

```
python main.py --sims 1000
```

Save outputs:

```
python main.py --save
```

---

## Running in notebook (Deepnote / Jupyter)

Typical workflow:

1. load data
2. build features
3. train qualifying model
4. generate qualifying predictions
5. simulate each race
6. compute standings
7. export results

---

## Project structure

```
f1-project/
├── f1_prediction.ipynb
├── data_loader.py
├── feature_engineering.py
├── quali_model.py
├── race_simulation.py
├── standings.py
├── main.py
├── requirements.txt
├── README.md
├── f1_cleaned_output/
│   ├── master/
│   └── priors/
├── 2026 cleaned/
└── deepnote_output/
```

---

## Exported outputs

Typical outputs include:

* `all_qualifying_predictions.csv`
* `all_race_summaries.csv`
* `all_race_probabilities.csv`
* `driver_standings.csv`
* `constructor_standings.csv`

---

## Modeling approach

* 2026 regulation changes reduce reliability of historical team performance
* current-season pace signals are prioritized
* driver skill transfers more consistently than car performance
* results are probabilistic rather than deterministic

---

## Notes

* Bahrain and Saudi Arabia were excluded from the simulation as they were canceled in the current season
* race outcomes are based on repeated simulations
* outputs represent probability distributions, not exact predictions

---

## Summary

This project builds a full-season Formula 1 simulation system using machine learning and probabilistic modeling to generate realistic race outcomes and championship projections.
