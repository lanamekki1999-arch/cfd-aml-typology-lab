"""
CFD Brokerage AML Typology Lab — synthetic data generator
-----------------------------------------------------------
Generates a fictional book of CFD brokerage customers, deposits, trades,
and withdrawals, with a KNOWN set of planted typology cases so detection
rules can be scored against ground truth (precision/recall), not vibes.

NO real customer, employer, or client data is used anywhere in this file.
All names, accounts and transactions are randomly generated.
"""

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

FIRST_NAMES = ["Ahmed", "Mohammed", "Fatima", "Sara", "Omar", "Layla", "Youssef",
               "Mariam", "Khalid", "Noor", "Hassan", "Aisha", "Tariq", "Huda",
               "James", "Olivia", "Wei", "Mei", "Ivan", "Elena", "Carlos", "Sofia",
               "Raj", "Priya", "Chen", "Anna", "Michael", "Jessica", "David", "Lina"]
LAST_NAMES = ["Al Mazrouei", "Hassan", "Khan", "Smith", "Zhang", "Petrov",
              "Garcia", "Patel", "Ibrahim", "Osman", "Chen", "Johnson",
              "Al Suwaidi", "Mansour", "Novak", "Silva", "Kumar", "Ali"]
COUNTRIES = ["UAE", "UK", "India", "Pakistan", "Nigeria", "Russia", "China",
             "Philippines", "Egypt", "Lebanon", "Ukraine", "South Africa",
             "Iran", "Syria"]  # a mix of standard and higher-risk onboarding geographies
HIGH_RISK_COUNTRIES = {"Iran", "Syria", "Russia"}  # purely illustrative, demo-only risk flag
BANK_COUNTRIES = COUNTRIES + ["Cayman Islands", "Panama", "Seychelles"]
INSTRUMENTS = ["EURUSD", "GBPUSD", "XAUUSD", "US30", "NAS100", "BTCUSD", "GER40"]

N_CUSTOMERS = 400
START_DATE = datetime(2026, 1, 1)
DAYS_RANGE = 240


def rand_date(start=START_DATE, days=DAYS_RANGE):
    return start + timedelta(days=random.randint(0, days), hours=random.randint(0, 23))


def gen_customers(n=N_CUSTOMERS):
    rows = []
    for i in range(n):
        country = random.choice(COUNTRIES)
        rows.append({
            "customer_id": f"C{i+1:04d}",
            "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "onboarding_country": country,
            "onboarding_date": rand_date(START_DATE - timedelta(days=180), 180),
            "account_type": random.choice(["Retail", "Retail", "Retail", "Professional"]),
        })
    return pd.DataFrame(rows)


def gen_normal_activity(customers):
    """Background 'normal' noise: ordinary depositing, trading, withdrawing customers."""
    deposits, trades, withdrawals = [], [], []
    dep_id = tr_id = wd_id = 1
    for _, cust in customers.iterrows():
        n_dep = np.random.poisson(3) + 1
        for _ in range(n_dep):
            amt = round(np.random.lognormal(mean=7.2, sigma=1.0), 2)  # ~$500-$20k typical
            deposits.append({
                "deposit_id": f"D{dep_id:05d}", "customer_id": cust.customer_id,
                "date": rand_date(), "amount_usd": amt,
                "source_country": cust.onboarding_country,  # matches onboarding = normal
                "funding_name_match": True,
            })
            dep_id += 1
        n_tr = np.random.poisson(25) + 1
        for _ in range(n_tr):
            trades.append({
                "trade_id": f"T{tr_id:06d}", "customer_id": cust.customer_id,
                "date": rand_date(), "instrument": random.choice(INSTRUMENTS),
                "volume_usd": round(np.random.lognormal(mean=6.5, sigma=1.0), 2),
                "counterparty_customer_id": None,
            })
            tr_id += 1
        n_wd = np.random.poisson(2)
        for _ in range(n_wd):
            withdrawals.append({
                "withdrawal_id": f"W{wd_id:05d}", "customer_id": cust.customer_id,
                "date": rand_date(), "amount_usd": round(np.random.lognormal(mean=7.0, sigma=0.9), 2),
                "destination_country": cust.onboarding_country,
            })
            wd_id += 1
    return (pd.DataFrame(deposits), pd.DataFrame(trades), pd.DataFrame(withdrawals),
            dep_id, tr_id, wd_id)


def plant_typology_cases(customers, deposits, trades, withdrawals, dep_id, tr_id, wd_id):
    """Inject ~5 clean examples of each typology, tagged with ground truth labels."""
    labels = {}  # customer_id -> list of typology labels (ground truth)
    cust_ids = list(customers.customer_id)

    def label(cid, tag):
        labels.setdefault(cid, []).append(tag)

    # 1. Deposit -> minimal trade -> withdraw (layering / pass-through)
    for cid in random.sample(cust_ids, 6):
        d = rand_date()
        deposits.loc[len(deposits)] = [f"D{dep_id:05d}", cid, d, round(random.uniform(8000, 40000), 2),
                                        customers.loc[customers.customer_id == cid, "onboarding_country"].values[0], True]
        dep_id += 1
        for _ in range(2):  # trivial trading only
            trades.loc[len(trades)] = [f"T{tr_id:06d}", cid, d + timedelta(hours=random.randint(1, 6)),
                                        random.choice(INSTRUMENTS), round(random.uniform(10, 50), 2), None]
            tr_id += 1
        withdrawals.loc[len(withdrawals)] = [f"W{wd_id:05d}", cid, d + timedelta(days=random.randint(1, 3)),
                                              round(random.uniform(7500, 39000), 2),
                                              customers.loc[customers.customer_id == cid, "onboarding_country"].values[0]]
        wd_id += 1
        label(cid, "pass_through_layering")

    # 2. Third-party funding mismatch
    for cid in random.sample(cust_ids, 6):
        other_country = random.choice([c for c in COUNTRIES if c != customers.loc[customers.customer_id == cid, "onboarding_country"].values[0]])
        deposits.loc[len(deposits)] = [f"D{dep_id:05d}", cid, rand_date(), round(random.uniform(3000, 25000), 2),
                                        other_country, False]
        dep_id += 1
        label(cid, "third_party_funding")

    # 3. Structuring: 4-6 deposits just under a $10k review threshold in <10 days
    for cid in random.sample(cust_ids, 6):
        base = rand_date()
        for k in range(random.randint(4, 6)):
            deposits.loc[len(deposits)] = [f"D{dep_id:05d}", cid, base + timedelta(days=k),
                                            round(random.uniform(9000, 9900), 2),
                                            customers.loc[customers.customer_id == cid, "onboarding_country"].values[0], True]
            dep_id += 1
        label(cid, "structuring")

    # 4. Linked-account wash trading (value transfer between two parties via opposite trades)
    pairs = random.sample(cust_ids, 12)
    for i in range(0, 12, 2):
        a, b = pairs[i], pairs[i + 1]
        for _ in range(5):
            d = rand_date()
            instr = random.choice(INSTRUMENTS)
            vol = round(random.uniform(5000, 20000), 2)
            trades.loc[len(trades)] = [f"T{tr_id:06d}", a, d, instr, vol, b]
            tr_id += 1
            trades.loc[len(trades)] = [f"T{tr_id:06d}", b, d + timedelta(minutes=random.randint(1, 15)),
                                        instr, vol, a]
            tr_id += 1
        label(a, "linked_wash_trading")
        label(b, "linked_wash_trading")

    # 5. Rapid jurisdiction-mismatch withdrawal after a deposit, minimal trading in between
    for cid in random.sample(cust_ids, 6):
        home = customers.loc[customers.customer_id == cid, "onboarding_country"].values[0]
        diff_country = random.choice([c for c in BANK_COUNTRIES if c != home])
        d = rand_date()
        deposits.loc[len(deposits)] = [f"D{dep_id:05d}", cid, d, round(random.uniform(10000, 30000), 2), home, True]
        dep_id += 1
        trades.loc[len(trades)] = [f"T{tr_id:06d}", cid, d + timedelta(hours=3), random.choice(INSTRUMENTS),
                                    round(random.uniform(20, 80), 2), None]
        tr_id += 1
        withdrawals.loc[len(withdrawals)] = [f"W{wd_id:05d}", cid, d + timedelta(hours=20),
                                              round(random.uniform(9500, 29500), 2), diff_country]
        wd_id += 1
        label(cid, "jurisdiction_mismatch_withdrawal")

    gt = pd.DataFrame([{"customer_id": cid, "typologies": ";".join(tags)} for cid, tags in labels.items()])
    return deposits, trades, withdrawals, gt


def build_dataset():
    customers = gen_customers()
    deposits, trades, withdrawals, dep_id, tr_id, wd_id = gen_normal_activity(customers)
    deposits, trades, withdrawals, ground_truth = plant_typology_cases(
        customers, deposits, trades, withdrawals, dep_id, tr_id, wd_id)
    return customers, deposits, trades, withdrawals, ground_truth


if __name__ == "__main__":
    customers, deposits, trades, withdrawals, ground_truth = build_dataset()
    customers.to_csv("customers.csv", index=False)
    deposits.to_csv("deposits.csv", index=False)
    trades.to_csv("trades.csv", index=False)
    withdrawals.to_csv("withdrawals.csv", index=False)
    ground_truth.to_csv("ground_truth.csv", index=False)
    print(f"customers={len(customers)} deposits={len(deposits)} trades={len(trades)} "
          f"withdrawals={len(withdrawals)} planted_cases={len(ground_truth)}")
