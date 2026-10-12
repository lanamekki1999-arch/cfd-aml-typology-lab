# CFD Brokerage AML Typology Lab

A portfolio project detecting five account-level money-laundering typologies specific to
CFD/FX brokerages and trading fintechs — the kind of pattern a KYC/AML analyst or
compliance manager at a brokerage actually investigates, not a generic sanctions-list demo.

**Built by Lana Mohammed** — based on AML/KYC/CDD/EDD and onboarding experience at a
DFSA-regulated CFD brokerage. All data in this project is synthetic. No real customer,
employer, or client information is used anywhere.

## Why this project

Most "AML + AI" portfolio pieces wrap a prompt around a sanctions list. A CFD brokerage's
actual exposure looks different: accounts that deposit, barely trade, and withdraw; funding
that doesn't match the account holder; deposits clustered just under a review threshold;
paired accounts quietly transferring value through opposite trades instead of a bank
transfer; and withdrawals that land in a different country from where the customer
onboarded. This project encodes those five patterns as configurable, testable rules.

## What it does

- Generates a synthetic book of 400 customers with realistic deposit/trade/withdrawal activity
- Plants 36 known typology cases across the five patterns (ground truth kept separate from detection logic)
- Runs five detection rules and scores the result against ground truth: **94.4% recall,
  87.2% precision, 9.75% alert rate** at default thresholds
- Streamlit app: adjustable thresholds, an alert queue with evidence per flag, and a
  customer-level disposition view

## Results (default thresholds)

| Metric | Value |
|---|---|
| True positives | 34 / 36 planted cases |
| False positives | 5 / 400 customers |
| Precision | 87.2% |
| Recall | 94.4% |
| Alert rate | 9.75% of book |

One documented iteration: the pass-through layering rule originally used a flat $200
trading-volume cap and missed a third of its own planted cases, because any unrelated
background trading in the detection window pushed volume over a fixed dollar cutoff.
Switching to a cap expressed as a percentage of the deposit fixed it. That's the kind of
thing you only catch by testing against known cases — which is the point of building it
this way rather than trusting the first version that runs.

## How it was built

1. Defined the five typologies from onboarding/transaction-monitoring experience, each
   mapped to a specific detectable pattern (not a vague "suspicious activity" label).
2. Built a synthetic data generator (Python, pandas/numpy) with realistic background
   "noise" activity plus deliberately planted cases with known labels.
3. Wrote the detection rules, then validated every rule against the planted ground truth
   — computing precision/recall rather than eyeballing the output.
4. Found and fixed a real logic bug (the flat-dollar threshold above) using that validation.
5. Built a Streamlit interface: live threshold sliders, an evidence-based alert queue, and
   a per-customer disposition view, so the output is something an analyst could actually work from.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud) by pointing it
at this repo — no server setup needed.

## Limitations (stated up front)

- Thresholds are illustrative defaults, not calibrated to any firm's EWRA or risk appetite — recalibrate before any real use.
- Precision/recall are measured against synthetic planted cases, which is a controlled test, not validation on live data.
- No PEP, sanctions, or adverse-media screening is included — this complements, not replaces, screening.
- This is a demonstration project, not legal advice, and not production transaction monitoring software.

## Stack

Python, pandas, numpy, Streamlit — no paid tools, no external data dependencies.
