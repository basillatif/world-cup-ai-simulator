# World Cup AI Simulator

A World Cup prediction engine combining a Monte Carlo tournament simulator with a Claude-powered analyst layer.

> **Core design principle:** GenAI is not replacing the model. It is explaining the model.

---

## How it works

```
teams.csv + matches.csv
        │
        ▼
  ELO Rating System ──┐
                       ├──► MatchPredictor (blended ensemble)
  Poisson Model ───────┘         │
                                 ▼
                    Monte Carlo Tournament Simulator
                         (10,000 iterations)
                                 │
                                 ▼
                       Probability Distributions
                                 │
                                 ▼
                    Claude Analyst Layer (explanation)
                                 │
                                 ▼
                          Streamlit Dashboard
```

**The statistical models do the prediction work.** Claude receives the model's numerical outputs — probabilities, expected goals, ELO differentials — and explains *why* the numbers look the way they do. Claude never guesses who will win; it narrates what the model found.

---

## Project structure

```
world-cup-ai-simulator/
├── data/
│   ├── raw/                        # Drop your own datasets here
│   ├── processed/
│   └── sample/
│       ├── teams.csv               # 32 teams with ELO, FIFA rank, form, squad value
│       ├── matches.csv             # Historical results (2021–2022)
│       └── groups.csv              # 2022 World Cup draw
├── src/
│   ├── data/load_data.py           # CSV loaders with validation
│   ├── features/build_features.py  # H2H, rolling form, matchup feature vectors
│   ├── models/
│   │   ├── elo.py                  # ELO rating system with goal-difference multiplier
│   │   ├── poisson_model.py        # Bivariate Poisson / Dixon-Coles model
│   │   └── match_predictor.py      # Ensemble blending ELO + Poisson
│   ├── simulation/
│   │   └── tournament_simulator.py # Full group + knockout Monte Carlo engine
│   ├── genai/
│   │   ├── analyst_agent.py        # Claude API integration
│   │   └── prompt_templates.py     # Structured prompts (data-in → narration-out)
│   └── app/
│       └── streamlit_app.py        # Interactive dashboard
└── tests/
    ├── test_elo.py
    └── test_simulator.py
```

---

## Quickstart

### 1. Install dependencies

```bash
cd world-cup-ai-simulator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your API key (optional — needed only for Claude commentary)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the Streamlit app

```bash
streamlit run src/app/streamlit_app.py
```

### 4. Run the tests

```bash
pytest
```

---

## Core modules

### ELO model (`src/models/elo.py`)

Implements the World Football Elo variant:
- K-factor varies by tournament importance (60 for World Cup, 20 for friendlies)
- Goal-difference multiplier (larger wins → larger rating updates)
- 100-point home advantage applied to non-neutral venues
- Win/draw/loss probability derived from expected score with a draw probability band

### Poisson model (`src/models/poisson_model.py`)

Dixon-Coles style attack/defense strength estimation:
- Each team has an `attack` and `defense` parameter
- Expected goals = `home_advantage × attack_home × defense_away`
- Parameters fit via maximum likelihood over historical scorelines
- Falls back to team-seeded values when historical data is sparse

### Match predictor (`src/models/match_predictor.py`)

Blends ELO (45%) and Poisson (55%) outcome probabilities. For simulated matches, scorelines are drawn from the Poisson distribution, with rejection sampling to ensure the scoreline is consistent with the blended outcome probability.

### Monte Carlo simulator (`src/simulation/tournament_simulator.py`)

Runs N full tournaments end-to-end:
1. Simulates all group-stage round-robin fixtures
2. Ranks teams by points → GD → GF (random tiebreak)
3. Simulates R16 → QF → SF → Final with coin-flip penalty shootouts for drawn knockout ties
4. Aggregates advancement probabilities across all simulations

### Claude analyst layer (`src/genai/`)

`analyst_agent.py` wraps the Claude API with four analysis modes:

| Method | What Claude receives | What Claude produces |
|---|---|---|
| `match_preview` | Win probs, xG, ELO diff, H2H, form | 3-4 paragraph match preview |
| `tournament_outlook` | MC stage probs, team profile | Tournament ceiling/floor analysis |
| `group_summary` | Group advance probs, team stats | Group narrative + dark-horse pick |
| `upset_alert` | Underdog win prob, ELO gap | 2-3 sentence upset context |

All prompts explicitly tell Claude the model has already made the prediction and its job is explanation only.

---

## Configuration

You can swap in your own data by passing paths to the loaders:

```python
from src.data.load_data import load_teams, load_matches, load_groups

teams = load_teams("data/raw/my_teams.csv")
matches = load_matches("data/raw/my_matches.csv")
groups = load_groups("data/raw/my_groups.csv")
```

To adjust the ELO/Poisson blend weight:

```python
from src.models.match_predictor import MatchPredictor
predictor = MatchPredictor(elo=elo, poisson=poisson, elo_weight=0.6, poisson_weight=0.4)
```

---

## Running a simulation programmatically

```python
from src.data.load_data import load_teams, load_matches, load_groups
from src.models.elo import build_elo_from_seed
from src.models.poisson_model import build_poisson_from_teams
from src.models.match_predictor import MatchPredictor
from src.simulation.tournament_simulator import run_monte_carlo

teams_df = load_teams()
matches_df = load_matches()
groups_df = load_groups()

elo = build_elo_from_seed(teams_df, matches_df)
poisson = build_poisson_from_teams(teams_df)
poisson.fit(matches_df)

predictor = MatchPredictor(elo=elo, poisson=poisson)

results = run_monte_carlo(groups_df, predictor, n_simulations=10_000)

for team, p in results["top_contenders"]:
    print(f"{team:20s} win: {p['champion']:.1%}  final: {p['final']:.1%}")
```
