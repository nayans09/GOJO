# GOJO

GOJO is a reproducible benchmark for studying how LLM teams use distributed evidence in three treasure-search scenarios:

1. direct target evidence;
2. reliable indirect glow evidence;
3. indirect evidence with decoy glows.

The primary comparison is between `shared_memory` coordination and `deliberation` (private observation, named peer reports, then a final vote). Episodes are independent, seeded, and stored without pooling scenario results.

## Setup

Create and activate a virtual environment, then install the dependency:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set `GOJO_API_KEY` in `.env`. Do not commit this file.

## Run a no-cost smoke test

```bash
python POC.py --dry-run --games 2 --all-scenarios --protocol deliberation --seed 20260730
```

## Run a benchmark cell

```bash
python POC.py --games 200 --scenario 3 --protocol deliberation --seed 20260730
python POC.py --games 200 --scenario 3 --protocol shared_memory --seed 20260730
```

Use the same seed set for both protocols. Each run gets an immutable manifest and resumable JSONL output directory. Re-run with `--run-id <existing-id>` only to resume the exact same configuration. Invalid episodes are retained and never silently replaced; plan additional seeds if you require a fixed number of valid episodes.

## Analyze one run

```bash
python analyze_gojo.py results/<run-id>
```

The analyzer reports task outcomes, initial/final accuracy, revision outcomes, calibration, agreement, decoy pursuit, cost, retries, and exploratory allocation fields. It does not make causal claims about peer influence; use preregistered targeted follow-up studies for those claims.

`analysis.py`, `analysis_export.py`, and `graphing.py` are legacy analyses for the pre-v2 log schema. Preserve them for historical data, but do not use them to analyze new `results/<run-id>/episodes.jsonl` files.
