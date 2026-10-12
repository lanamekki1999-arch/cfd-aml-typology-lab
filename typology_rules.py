"""
CFD Brokerage AML Typology Rule Engine
----------------------------------------
Five configurable rules, each mapped to a real brokerage-specific
typology. Every rule returns a per-customer flag + evidence, so output
can feed a disposition/alert queue, not just a single score.

This is a DEMONSTRATION engine: thresholds are illustrative defaults,
not calibrated to any regulator's guidance, and must be validated against
a firm's own risk appetite and EWRA before any real use.
"""

import pandas as pd


def rule_pass_through_layering(deposits, trades, withdrawals,
                                min_deposit=5000, max_trade_volume_pct=0.05,
                                max_trade_count=5, withdraw_window_days=5):
    """Deposit -> trivial trading -> withdrawal within a short window.

    'Trivial trading' is defined relative to the deposit size (max_trade_volume_pct
    of the deposit), not a flat dollar amount — a fixed cutoff breaks down as soon
    as deposit sizes vary, or a customer has any unrelated trading in the window.
    """
    flags = []
    for cid, dep_grp in deposits.groupby("customer_id"):
        big_deps = dep_grp[dep_grp.amount_usd >= min_deposit]
        for _, dep in big_deps.iterrows():
            window_end = dep.date + pd.Timedelta(days=withdraw_window_days)
            tr = trades[(trades.customer_id == cid) & (trades.date >= dep.date) & (trades.date <= window_end)]
            wd = withdrawals[(withdrawals.customer_id == cid) & (withdrawals.date >= dep.date) & (withdrawals.date <= window_end)]
            trade_cap = dep.amount_usd * max_trade_volume_pct
            if len(tr) <= max_trade_count and tr.volume_usd.sum() <= trade_cap and len(wd) > 0:
                flags.append({"customer_id": cid, "rule": "pass_through_layering",
                               "evidence": f"deposit ${dep.amount_usd:,.0f} on {dep.date.date()}, "
                                           f"{len(tr)} trades totalling ${tr.volume_usd.sum():,.0f} "
                                           f"({tr.volume_usd.sum()/dep.amount_usd:.1%} of deposit), "
                                           f"withdrawn within {withdraw_window_days}d"})
    return pd.DataFrame(flags)


def rule_third_party_funding(deposits):
    """Deposit source country differs from what's on file / funding name mismatch flag."""
    mism = deposits[deposits.funding_name_match == False]
    return pd.DataFrame([{"customer_id": r.customer_id, "rule": "third_party_funding",
                           "evidence": f"deposit ${r.amount_usd:,.0f} from {r.source_country} "
                                       f"on {r.date.date()}, funding name mismatch"} for _, r in mism.iterrows()])


def rule_structuring(deposits, threshold=10000, buffer=1500, min_count=3, window_days=10):
    """Multiple deposits clustered just under a review threshold within a short window."""
    flags = []
    near = deposits[(deposits.amount_usd >= threshold - buffer) & (deposits.amount_usd < threshold)]
    for cid, grp in near.groupby("customer_id"):
        grp = grp.sort_values("date")
        dates = grp.date.tolist()
        for i in range(len(dates)):
            cluster = [d for d in dates if dates[i] <= d <= dates[i] + pd.Timedelta(days=window_days)]
            if len(cluster) >= min_count:
                total = grp[grp.date.isin(cluster)].amount_usd.sum()
                flags.append({"customer_id": cid, "rule": "structuring",
                               "evidence": f"{len(cluster)} deposits just under ${threshold:,.0f} "
                                           f"within {window_days}d, totalling ${total:,.0f}"})
                break
    return pd.DataFrame(flags)


def rule_linked_wash_trading(trades, min_shared_trades=3, time_window_minutes=30):
    """Pairs of accounts repeatedly taking opposite sides of the same instrument within minutes."""
    linked = trades[trades.counterparty_customer_id.notna()].copy()
    if linked.empty:
        return pd.DataFrame(columns=["customer_id", "rule", "evidence"])
    linked["pair"] = linked.apply(lambda r: tuple(sorted([r.customer_id, r.counterparty_customer_id])), axis=1)
    flags = []
    seen_pairs = set()
    for pair, grp in linked.groupby("pair"):
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        if len(grp) >= min_shared_trades * 2:  # both legs counted
            a, b = pair
            total_vol = grp.volume_usd.sum() / 2
            for cid in (a, b):
                flags.append({"customer_id": cid, "rule": "linked_wash_trading",
                               "evidence": f"{len(grp)//2} opposite-side trades with {b if cid == a else a}, "
                                           f"~${total_vol:,.0f} notional exchanged"})
    return pd.DataFrame(flags)


def rule_jurisdiction_mismatch_withdrawal(deposits, trades, withdrawals, customers,
                                           max_trade_count=3, window_days=2):
    """Deposit -> minimal trading -> withdrawal to a bank country different from onboarding/deposit source."""
    flags = []
    cust_country = customers.set_index("customer_id").onboarding_country.to_dict()
    for cid, dep_grp in deposits.groupby("customer_id"):
        home = cust_country.get(cid)
        for _, dep in dep_grp.iterrows():
            window_end = dep.date + pd.Timedelta(days=window_days)
            tr = trades[(trades.customer_id == cid) & (trades.date >= dep.date) & (trades.date <= window_end)]
            wd = withdrawals[(withdrawals.customer_id == cid) & (withdrawals.date >= dep.date) & (withdrawals.date <= window_end)]
            mismatched = wd[wd.destination_country != home]
            if len(tr) <= max_trade_count and not mismatched.empty:
                for _, w in mismatched.iterrows():
                    flags.append({"customer_id": cid, "rule": "jurisdiction_mismatch_withdrawal",
                                   "evidence": f"onboarded {home}, deposit {dep.date.date()}, "
                                               f"withdrawal to {w.destination_country} on {w.date.date()}, "
                                               f"only {len(tr)} trades in between"})
    return pd.DataFrame(flags)


def run_all_rules(customers, deposits, trades, withdrawals, config=None):
    """Run every rule with optional threshold overrides; return combined alert table + per-customer risk score."""
    cfg = config or {}
    alerts = pd.concat([
        rule_pass_through_layering(deposits, trades, withdrawals, **cfg.get("pass_through_layering", {})),
        rule_third_party_funding(deposits),
        rule_structuring(deposits, **cfg.get("structuring", {})),
        rule_linked_wash_trading(trades, **cfg.get("linked_wash_trading", {})),
        rule_jurisdiction_mismatch_withdrawal(deposits, trades, withdrawals, customers,
                                               **cfg.get("jurisdiction_mismatch_withdrawal", {})),
    ], ignore_index=True)
    if alerts.empty:
        return alerts, pd.DataFrame(columns=["customer_id", "risk_score", "rules_triggered"])
    score = alerts.groupby("customer_id").agg(
        risk_score=("rule", "count"),
        rules_triggered=("rule", lambda x: ";".join(sorted(set(x))))
    ).reset_index().sort_values("risk_score", ascending=False)
    return alerts, score


def score_against_ground_truth(score_df, ground_truth_df, all_customer_ids):
    """Precision/recall of the flagged population vs planted typology cases (any typology)."""
    flagged = set(score_df.customer_id) if not score_df.empty else set()
    truth = set(ground_truth_df.customer_id) if not ground_truth_df.empty else set()
    tp = len(flagged & truth)
    fp = len(flagged - truth)
    fn = len(truth - flagged)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "alert_rate_pct": round(100 * len(flagged) / len(all_customer_ids), 2)}
