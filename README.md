# F1 Championship Prediction

Who will win the Formula 1 world championship, and how sure can we be after each race?

This project answers that in two ways and checks both against twelve past seasons:

- **A season simulation** (the headline model): play out the remaining races thousands of times and count how often each driver and team finishes on top.
- **An LSTM classifier** (PyTorch), kept as a comparison: a neural network that reads each driver's season so far and outputs a title probability.

The standings are rebuilt every Monday from published race results by a GitHub Action, and the sections marked as generated below are rendered from results files. A test (run in CI) fails if the README and those files disagree.

## Current prediction

<!-- prediction:start -->
**2026 season after round 21 of 23 (Las Vegas Grand Prix).** 2 rounds left; a driver can still score at most 50 points.

**ANT has won the 2026 drivers' championship**: the lead is larger than the points still available.

| Driver | Team | Points | Title probability |
|---|---|---|---|
| ANT | Mercedes | 402 | 100.0% |
| VER | Red Bull | 304 | 0.0% |
| HAM | Ferrari | 267 | 0.0% |
| NOR | McLaren | 264 | 0.0% |
| RUS | Mercedes | 264 | 0.0% |
| LEC | Ferrari | 250 | 0.0% |

| Team | Points | Constructors' title probability |
|---|---|---|
| Mercedes | 666 | 100.0% |
| Ferrari | 517 | 0.0% |
| Red Bull | 440 | 0.0% |
| McLaren | 431 | 0.0% |

From 20,000 simulated seasons. Only the top six drivers and top four teams are shown.
<!-- prediction:end -->

![Current prediction](results/2026_prediction.png)

## How good are the predictions?

Each method was replayed through past seasons, using only what was known after each round.

<!-- backtest:start -->
Seasons 2014–2025, one prediction after every round from round 4 to the second-last: 181 predictions per method. A further 23 rounds where the title was already decided by arithmetic are left out, since every method gets those right. The simulation uses 5,000 runs per prediction here.

| All seasons | Top pick was the champion | Probability given to the champion | Brier score (lower is better) |
|---|---|---|---|
| Season simulation | 72.4% | 71.7% | 0.394 |
| Points leader, hedged | 74.0% | 60.2% | 0.415 |
| Points leader, certain | 74.0% | 74.0% | 0.519 |
| LSTM classifier | 54.1% | 48.0% | 0.489 |

**Is the simulation as sure as it should be?** When its top pick had this much probability, this is how often the pick went on to win:

| Simulation's confidence in its pick | Predictions | Average confidence | Pick was right |
|---|---|---|---|
| 50–70% | 33 | 60% | 48% |
| 70–90% | 40 | 82% | 57% |
| 90–99% | 35 | 94% | 69% |
| 99–100% | 73 | 100% | 93% |

By stage of the season (Brier score, lower is better):

| Stage | Predictions | Season simulation | Points leader, hedged | Points leader, certain | LSTM classifier |
|---|---|---|---|---|---|
| First third | 45 | 0.457 | 0.466 | 0.622 | 0.598 |
| Middle third | 83 | 0.483 | 0.500 | 0.675 | 0.492 |
| Final third | 53 | 0.202 | 0.237 | 0.189 | 0.391 |

By season (how often the top pick was the eventual champion):

| Season | Champion | Season simulation | Points leader, hedged | Points leader, certain | LSTM classifier |
|---|---|---|---|---|---|
| 2014 | HAM | 40% | 40% | 40% | 0% |
| 2015 | HAM | 100% | 100% | 100% | 100% |
| 2016 | ROS | 76% | 76% | 76% | 100% |
| 2017 | HAM | 36% | 36% | 36% | 0% |
| 2018 | HAM | 80% | 80% | 80% | 20% |
| 2019 | HAM | 93% | 93% | 93% | 67% |
| 2020 | HAM | 100% | 100% | 100% | 70% |
| 2021 | VER | 67% | 78% | 78% | 28% |
| 2022 | VER | 86% | 86% | 86% | 0% |
| 2023 | VER | 100% | 100% | 100% | 100% |
| 2024 | VER | 100% | 100% | 100% | 100% |
| 2025 | NOR | 20% | 25% | 25% | 65% |
<!-- backtest:end -->

**What this shows.**

- **Picking the champion:** nothing beats the simplest rule, "whoever leads the standings wins". The simulation's top pick is nearly always the leader too, so it is right about as often.
- **Being appropriately sure:** the fair comparison is the hedged leader rule, which backs the leader only as strongly as that rule has been right in other seasons. The simulation's Brier score is slightly better than that overall and in each third of a season, though in the final third simply backing the leader with certainty scores best. Twelve seasons is too few to call any of these gaps a real difference.
- **The simulation is overconfident.** The calibration table shows it: picks it gave 70–90% went on to win far less often than that, and it has been close to certain about the wrong driver (Leclerc after four rounds of 2022, Piastri late in 2025). Treat its probabilities as too extreme, and treat a 0% as "would need to outscore anything seen so far this season", not as impossible.
- **The LSTM** picks the champion less often than either leader rule. Its Brier score is better than the certain leader rule and worse than the hedged one and the simulation. It was trained on 2014–2024 and 2025 was used to choose which training run to keep, so none of these seasons is new to it, which makes that comparison generous to it.
- **Late lead changes beat everything** (2014, 2017, 2021, 2025). Nothing here models car development or swings in form.

**The scores.** "Top pick was the champion" is plain accuracy. "Probability given to the champion" is how much weight the method put on the driver who went on to win. The Brier score is the squared error of the whole probability table: 0 is a perfect call and 2 is a certain wrong one.

## How the simulation works

Each remaining round is filled in with a round already run this season, drawn at random. Every driver and team gets the points they really scored in the drawn round. Do this for every remaining round, add the points to the current standings, and note who finishes on top. Repeat 20,000 times.

What this design gets right that a per-driver classifier cannot:

- **Probabilities add up to 100%**, because each simulated season has exactly one champion.
- **Points within a round stay consistent.** Whole rounds are drawn, so two drivers cannot both win the same race.
- **A driver who is out of reach gets 0%**, because nobody can score more in a simulated round than someone really scored in a round this season. Separately from the simulation, the report states the plain arithmetic: once the lead is larger than the points still available under that season's rules, the title is decided.
- **Sprint weekends are drawn from past sprint weekends** when there are any, because more points are on offer.
- **Team points follow the team a driver was with at the time**, so mid-season driver changes do not move points between teams.

### Limitations

- **It is overconfident**, as the calibration table shows. Redrawing past rounds cannot produce anything that has not already happened this season, so a driver or team that improves later is invisible to it.
- **Early in a season it has almost nothing to draw from.** No probabilities are published before round 6; the standings are shown instead.
- It knows nothing about upgrades, driver changes or which circuits remain.
- Drivers level on points are split at random, and standings ties are ordered by wins only; the real tie-break is a full countback.
- A driver's results are not separated from their car's. A model of driver and car strength with proper uncertainty is the next step.

## The LSTM classifier

```
Input per round (12 features): points and position (normalised), gap to leader,
win rate, podium rate, points per race, season progress, consistency,
momentum, is-leader flag, points share, recent form
        ↓
2-layer LSTM (hidden size 64, dropout 0.3)
        ↓
Linear → sigmoid: "does this driver win the title?"
```

53,313 parameters, trained with a class-weighted binary loss on 2014–2024, with 2025 used for early stopping.

Its design has a built-in weakness for this problem: each driver is scored on their own, so the outputs do not add up to one and the model cannot see that a rival is out of reach. The numbers in the backtest table are its outputs rescaled to sum to one.

The checkpoint in `models/` was trained on an earlier version of the dataset that left out sprint points. It has not been retrained on the corrected data, which is one more reason to treat it as a comparison and not as the model of record. `results/training_curves.png` and `results/model_architecture.png` are from that training run.

## Data

Race results, sprint results and calendars for 2014–2026 come from the [Jolpica API](https://api.jolpi.ca) (the successor to Ergast) and are stored in `data/season_results.csv` and `data/calendar.csv`.

`data/championship_standings.csv` is always rebuilt from the full list of results. It is never updated one round at a time, so a driver who misses a round keeps their points and a corrected result upstream corrects everything downstream.

## Running it

```bash
pip install -r requirements.txt

pytest                              # run the tests
python src/update_predictions.py    # refresh this season, re-simulate, update README and chart
python -m src.backtest              # replay 2014–2025 and write results/backtest.json
python src/train.py                 # retrain the LSTM (optional)
```

The prediction service in `api/` (FastAPI, with a Dockerfile) serves the LSTM classifier's raw outputs. It predates the simulation and has not been updated to use it.

## Project structure

```
f1-championship-prediction/
├── src/
│   ├── results_data.py        # official results and calendar
│   ├── standings.py           # standings rebuilt from results
│   ├── season_simulation.py   # rest-of-season simulation
│   ├── backtest.py            # replay past seasons, score each method
│   ├── report.py              # prediction file and generated README blocks
│   ├── update_predictions.py  # weekly update entry point
│   ├── points.py              # points available under each season's rules
│   └── model.py, dataset.py, feature_engineering.py, train.py   # LSTM
├── data/                      # results, calendar, standings
├── results/                   # prediction.json, backtest.json, charts
├── api/                       # FastAPI service for the LSTM
├── notebooks/                 # original LSTM analysis notebook
└── tests/
```

The notebook and the constructor LSTM scripts (`train_constructor.py`, `constructor_feature_engineering.py`) are from the original LSTM-only version of the project and are not part of the weekly update. The constructors' prediction now comes from the same simulation as the drivers'.

## Future work

- A driver-and-car strength model with uncertainty, in place of redrawing past rounds.
- Retrain the LSTM on the corrected data with each season held out in turn, so its backtest is a fair one.
- Countback on wins for ties.
