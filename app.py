"""
CFD Brokerage AML Typology Lab — Streamlit app
A portfolio demo: synthetic-data typology detection for CFD brokerages / fintechs.
NOT legal advice. NOT production screening. Synthetic data only.
"""
import streamlit as st
import pandas as pd
from data_generator import build_dataset
from typology_rules import run_all_rules, score_against_ground_truth

st.set_page_config(page_title="CFD Brokerage AML Typology Lab", layout="wide")

st.title("CFD Brokerage AML Typology Lab")
st.caption(
    "A synthetic-data demo detecting five account-level laundering typologies specific to "
    "CFD/FX brokerages and trading fintechs — built from onboarding, deposit, trade and "
    "withdrawal data, not sanctions-list matching. **Demo only — synthetic data, "
    "illustrative thresholds, not legal or regulatory advice.**"
)

@st.cache_data
def load_data():
    customers, deposits, trades, withdrawals, ground_truth = build_dataset()
    for df, col in [(deposits, "date"), (trades, "date"), (withdrawals, "date"), (customers, "onboarding_date")]:
        df[col] = pd.to_datetime(df[col])
    return customers, deposits, trades, withdrawals, ground_truth

customers, deposits, trades, withdrawals, ground_truth = load_data()

st.sidebar.header("Rule thresholds")
st.sidebar.caption("All thresholds are illustrative — adjust and watch precision/recall change live.")

min_deposit = st.sidebar.slider("Pass-through: min deposit (USD)", 1000, 20000, 5000, step=500)
max_trade_pct = st.sidebar.slider("Pass-through: max trading as % of deposit", 1, 30, 5) / 100
withdraw_window = st.sidebar.slider("Pass-through: withdrawal window (days)", 1, 14, 5)
struct_threshold = st.sidebar.slider("Structuring: review threshold (USD)", 5000, 20000, 10000, step=500)
struct_buffer = st.sidebar.slider("Structuring: buffer below threshold (USD)", 500, 3000, 1500, step=100)
struct_min_count = st.sidebar.slider("Structuring: min deposits in window", 2, 8, 3)
wash_min_trades = st.sidebar.slider("Wash trading: min opposite-side trades (per leg)", 2, 10, 3)

config = {
    "pass_through_layering": {"min_deposit": min_deposit, "max_trade_volume_pct": max_trade_pct,
                               "withdraw_window_days": withdraw_window},
    "structuring": {"threshold": struct_threshold, "buffer": struct_buffer, "min_count": struct_min_count},
    "linked_wash_trading": {"min_shared_trades": wash_min_trades},
}

alerts, score = run_all_rules(customers, deposits, trades, withdrawals, config=config)
metrics = score_against_ground_truth(score, ground_truth, customers.customer_id.tolist())

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Alert Queue", "Customer Detail", "Methodology"])

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", len(customers))
    c2.metric("Flagged", f"{score.customer_id.nunique() if not score.empty else 0}",
              f"{metrics['alert_rate_pct']}% of book")
    c3.metric("Precision*", f"{metrics['precision']:.0%}")
    c4.metric("Recall*", f"{metrics['recall']:.0%}")
    c5.metric("Total alerts", len(alerts))
    st.caption("*Measured against 36 planted ground-truth typology cases — not applicable to a real book "
               "with unknown ground truth. See Methodology tab.")

    if not alerts.empty:
        st.subheader("Alerts by typology")
        by_rule = alerts.rule.value_counts().rename_axis("typology").reset_index(name="alerts")
        st.bar_chart(by_rule.set_index("typology"))

    st.subheader("Why this is different from a generic sanctions-screening demo")
    st.markdown("""
- **Account-level, not list-matching.** These five typologies look at deposit → trade → withdrawal
  behaviour over time — the pattern a CFD brokerage actually sees, not a name on a list.
- **Every threshold is a slider**, because a real EWRA sets these per firm risk appetite, not a model author.
- **Scored against planted ground truth**, so precision/recall are measurable, not claimed.
- **Evidence, not a black-box score** — every alert row shows the deposit, trade and withdrawal
  facts a KYC/AML analyst would need to disposition it.
""")

with tab2:
    st.subheader("Alert queue")
    if alerts.empty:
        st.info("No alerts at current thresholds.")
    else:
        rule_filter = st.multiselect("Filter by typology", sorted(alerts.rule.unique()),
                                      default=sorted(alerts.rule.unique()))
        filtered = alerts[alerts.rule.isin(rule_filter)].merge(
            customers[["customer_id", "name", "onboarding_country", "account_type"]], on="customer_id")
        st.dataframe(filtered[["customer_id", "name", "onboarding_country", "account_type", "rule", "evidence"]],
                     use_container_width=True, height=450)
        st.download_button("Download alert queue (CSV)", filtered.to_csv(index=False),
                            "alert_queue.csv", "text/csv")

with tab3:
    st.subheader("Customer detail / disposition view")
    if not score.empty:
        cid = st.selectbox("Select a flagged customer", score.customer_id.tolist())
        cust = customers[customers.customer_id == cid].iloc[0]
        st.write(f"**{cust['name']}** — onboarded {cust.onboarding_country}, "
                 f"{cust.onboarding_date.date()}, {cust.account_type}")
        row = score[score.customer_id == cid].iloc[0]
        st.write(f"Risk score: **{row.risk_score}** | Rules triggered: {row.rules_triggered}")
        st.markdown("**Evidence:**")
        for _, a in alerts[alerts.customer_id == cid].iterrows():
            st.write(f"- `{a.rule}`: {a.evidence}")
        disposition = st.radio("Disposition (demo only — not saved)", ["Not reviewed", "Discounted — false positive", "Escalate for EDD/STR assessment"])
        rationale = st.text_area("Rationale")
        st.caption("In a production tool this would write to a case management log with a timestamp and reviewer ID.")
    else:
        st.info("No flagged customers at current thresholds.")

with tab4:
    st.markdown("""
### Methodology
Synthetic book of 400 fictional customers with realistic background deposit/trade/withdrawal
activity, plus 36 deliberately planted cases across five typologies (ground truth labels kept
separate from the detection logic). No real customer, employer, or client data is used anywhere.

### Typologies detected
1. **Pass-through layering** — deposit, trivial trading relative to deposit size, withdrawal within days.
2. **Third-party funding** — deposit source/name doesn't match the account holder on file.
3. **Structuring** — repeated deposits clustered just under a review threshold.
4. **Linked-account wash trading** — paired accounts taking opposite sides of the same instrument
   within minutes, transferring value without a bank transfer.
5. **Jurisdiction-mismatch withdrawal** — deposit, minimal trading, withdrawal to a bank country
   different from onboarding, in a short window.

### Known limitations (stated, not hidden)
- Thresholds are illustrative defaults and must be recalibrated against a firm's own EWRA and risk appetite.
- Precision/recall are measured against planted synthetic cases, which is a controlled test, not
  a validation on live data — a real book will have typologies this rule set doesn't cover.
- No PEP, sanctions, or adverse-media signal is included here — this engine complements, not replaces, screening.
- A fixed trade-volume cap breaks when trading activity varies by deposit size; this version uses a
  percentage-of-deposit cap instead, after testing showed the flat-dollar version missed ~33% of a
  planted typology in early iteration — kept here deliberately as an example of why every threshold
  needs testing against ground truth before trusting it.
""")
