"""
generate_data.py
----------------
Generates a fully synthetic, reproducible time-series dataset for the
ChurnAI LSTM model. Outputs two CSVs:

  1. data/customers.csv      — one row per customer (id, churn label, contract)
  2. data/time_series.csv    — N_CUSTOMERS x 12 rows (one row per customer-month)
  3. data/time_series_sample.csv — first 25 customers (for documentation use)

Behavioral signal design:

  • Retained customers (Churn=0):
      - Stable or slightly growing data usage, call minutes, and logins
      - 0–1 support tickets per month, near-zero outages
      - Monthly charge oscillates gently around a customer-specific base

  • Churners (Churn=1):
      - Months 1–6  : near-normal baseline
      - Months 7–10 : declining usage and logins, charges drift up
      - Months 11–12: sharp drop in usage and logins, 3x more tickets,
                      more service outages

Random seeds are derived from the customer ID, so the dataset is fully
reproducible across runs and machines.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hashlib
import numpy as np
import pandas as pd

import config as C


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seed_from_id(customer_id: str) -> int:
    """Deterministic per-customer seed so the dataset is reproducible."""
    h = hashlib.md5(customer_id.encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def _make_customer_ids(n: int) -> list[str]:
    """Create stable Telco-style IDs: e.g. '0001-ABCDE'."""
    rng = np.random.default_rng(C.RANDOM_SEED)
    letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    ids = []
    for i in range(n):
        suffix = "".join(rng.choice(letters, size=5))
        ids.append(f"{i+1:04d}-{suffix}")
    return ids


def _generate_customer_base() -> pd.DataFrame:
    """Generate the static customer table (id, churn label, contract type)."""
    rng = np.random.default_rng(C.RANDOM_SEED)

    ids = _make_customer_ids(C.N_CUSTOMERS)
    churn = rng.binomial(1, C.CHURN_RATE, size=C.N_CUSTOMERS)

    contract_choices = ["Month-to-month", "One year", "Two year"]
    # Month-to-month customers churn at much higher rates in reality:
    contract = np.where(
        churn == 1,
        rng.choice(contract_choices, size=C.N_CUSTOMERS, p=[0.78, 0.15, 0.07]),
        rng.choice(contract_choices, size=C.N_CUSTOMERS, p=[0.30, 0.35, 0.35]),
    )

    internet_choices = ["DSL", "Fiber optic", "No"]
    internet = np.where(
        churn == 1,
        rng.choice(internet_choices, size=C.N_CUSTOMERS, p=[0.20, 0.70, 0.10]),
        rng.choice(internet_choices, size=C.N_CUSTOMERS, p=[0.40, 0.40, 0.20]),
    )

    tenure_months = np.where(
        churn == 1,
        rng.integers(1, 30, size=C.N_CUSTOMERS),
        rng.integers(12, 72, size=C.N_CUSTOMERS),
    )

    return pd.DataFrame({
        "customerID":     ids,
        "churn":          churn.astype(int),
        "contract":       contract,
        "internetService": internet,
        "tenure_months":  tenure_months.astype(int),
    })


# ---------------------------------------------------------------------------
# Time-series generator
# ---------------------------------------------------------------------------
def _generate_sequence(customer_id: str, churn: int, contract: str) -> pd.DataFrame:
    """Generate a 12-month behavioral sequence for one customer.

    Each customer is given a per-customer noise level + a per-customer
    "ambiguity factor" so that ~20% of churners look mild and ~20% of
    retained customers look noisy. This produces a realistic class overlap
    (test AUC typically ~0.88–0.94) instead of a trivially separable
    problem.
    """
    rng = np.random.default_rng(_seed_from_id(customer_id))

    # Per-customer baselines
    base_data    = rng.uniform(12, 28)        # GB / month
    base_calls   = rng.uniform(110, 220)      # minutes / month
    base_charge  = rng.uniform(35, 95)        # $ / month
    base_logins  = rng.uniform(18, 32)        # logins / month

    # Per-customer noise multiplier — heavy tail so some customers are very noisy
    noise_mult = float(rng.uniform(1.0, 3.0))

    # Per-customer "ambiguity" — how mild/sharp this individual's pattern is
    # 0.0 = textbook example, 1.0 = pattern almost looks like the other class.
    # Beta(1.2, 1.4) is broad with mean ~0.46 → a third of customers in each
    # class will look genuinely ambiguous, producing realistic class overlap.
    ambiguity = float(rng.beta(1.2, 1.4))

    months = np.arange(1, C.SEQ_LEN + 1)

    if churn == 1:
        # ---------- Churner profile ----------
        # Sharpness of the decline scales inversely with ambiguity.
        # At ambiguity=0 → textbook 40-65% drop in last 3 months.
        # At ambiguity=1 → almost no decline at all.
        sharp_floor1 = 0.78 + 0.20 * ambiguity   # 0.78 (sharp) -> 0.98 (mild)
        sharp_floor2 = 0.50 + 0.45 * ambiguity   # 0.50 (sharp) -> 0.95 (mild)

        decline = np.ones(C.SEQ_LEN)
        decline[6:10] = np.linspace(0.94, sharp_floor1, 4)
        decline[10:]  = np.linspace(sharp_floor1 - 0.04, sharp_floor2, 2)

        data_gb       = base_data    * decline + rng.normal(0, 2.5 * noise_mult, C.SEQ_LEN)
        call_minutes  = base_calls   * decline + rng.normal(0, 18.0 * noise_mult, C.SEQ_LEN)
        login_count   = base_logins  * decline + rng.normal(0, 3.0 * noise_mult, C.SEQ_LEN)

        # Charges drift up but the magnitude varies a lot per customer
        drift_end = 1.0 + rng.uniform(-0.04, 0.12)  # some churners actually get a discount
        monthly_charge = (
            base_charge * np.linspace(1.0, drift_end, C.SEQ_LEN)
            + rng.normal(0, 3.0 * noise_mult, C.SEQ_LEN)
        )

        # Tickets — Poisson rate that is heavily dampened by ambiguity.
        # Ambiguous churners barely complain (≈ 0.5 tickets/month).
        end_rate = max(0.35, 2.5 * (1.0 - ambiguity))
        ticket_rate = np.concatenate([
            np.full(6, 0.30),
            np.linspace(0.40, end_rate * 0.5, 3),
            np.linspace(end_rate * 0.7, end_rate, 3),
        ])
        support_tickets = rng.poisson(ticket_rate).astype(float)

        end_out = max(0.25, 1.0 * (1.0 - ambiguity))
        outage_rate = np.concatenate([
            np.full(6, 0.20),
            np.full(3, 0.35),
            np.full(3, end_out),
        ])
        service_outages = rng.poisson(outage_rate).astype(float)

    else:
        # ---------- Retained customer profile ----------
        growth_end = 1.0 + rng.uniform(-0.08, 0.10)
        growth = np.linspace(1.0, growth_end, C.SEQ_LEN)

        # Roughly 35% of retained customers have a temporary dip that
        # could be mistaken for early-churn behaviour.
        if ambiguity > 0.55:
            dip_month  = int(rng.integers(3, 10))
            dip_len    = int(rng.integers(1, 4))           # 1-3 months
            dip_factor = 1.0 - rng.uniform(0.15, 0.30)
            growth[dip_month:dip_month + dip_len] *= dip_factor

        data_gb        = base_data    * growth + rng.normal(0, 2.0 * noise_mult, C.SEQ_LEN)
        call_minutes   = base_calls   * growth + rng.normal(0, 13.0 * noise_mult, C.SEQ_LEN)
        login_count    = base_logins  * growth + rng.normal(0, 2.8 * noise_mult, C.SEQ_LEN)
        monthly_charge = base_charge          + rng.normal(0, 2.5 * noise_mult, C.SEQ_LEN)

        # Some retained customers are heavy ticket openers — ambiguous ones
        # generate as many tickets as a mild churner.
        ticket_rate = np.full(C.SEQ_LEN, 0.20 + 0.9 * ambiguity)
        support_tickets = rng.poisson(ticket_rate).astype(float)

        outage_rate = np.full(C.SEQ_LEN, 0.15 + 0.5 * ambiguity)
        service_outages = rng.poisson(outage_rate).astype(float)

    # Clip to physically valid ranges
    data_gb        = np.clip(data_gb,        0.0, None)
    call_minutes   = np.clip(call_minutes,   0.0, None)
    monthly_charge = np.clip(monthly_charge, 0.0, None)
    login_count    = np.clip(login_count,    0.0, None)
    support_tickets = np.clip(support_tickets, 0.0, None)
    service_outages = np.clip(service_outages, 0.0, None)

    return pd.DataFrame({
        "customerID":      customer_id,
        "month":           months,
        "data_gb":         np.round(data_gb, 2),
        "call_minutes":    np.round(call_minutes, 1),
        "monthly_charge":  np.round(monthly_charge, 2),
        "support_tickets": support_tickets.astype(int),
        "login_count":     login_count.round().astype(int),
        "service_outages": service_outages.astype(int),
        "churn_label":     churn,
        "contract":        contract,
    })


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build customer base + 12-month time-series for every customer."""
    customers = _generate_customer_base()

    sequences = [
        _generate_sequence(row.customerID, int(row.churn), row.contract)
        for row in customers.itertuples(index=False)
    ]
    ts = pd.concat(sequences, ignore_index=True)

    customers.to_csv(C.CUSTOMERS_CSV, index=False)
    ts.to_csv(C.TIMESERIES_CSV, index=False)

    # Documentation-friendly sample: first 25 customers (= 300 rows)
    sample_ids = customers["customerID"].head(25).tolist()
    ts[ts["customerID"].isin(sample_ids)].to_csv(C.TIMESERIES_SAMPLE_CSV, index=False)

    return customers, ts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_preview(customers: pd.DataFrame, ts: pd.DataFrame) -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("─" * 72)
    print(f"Customers          : {len(customers):>6,}")
    print(f"Time-series rows   : {len(ts):>6,} "
          f"(= {len(customers)} customers x {C.SEQ_LEN} months)")
    print(f"Overall churn rate : {customers['churn'].mean():.1%}")
    print("─" * 72)

    churner = customers[customers["churn"] == 1].iloc[0]["customerID"]
    retained = customers[customers["churn"] == 0].iloc[0]["customerID"]

    print(f"\nCHURNER sample — {churner}")
    print(ts[ts["customerID"] == churner].drop(columns=["customerID", "contract"]).to_string(index=False))

    print(f"\nRETAINED sample — {retained}")
    print(ts[ts["customerID"] == retained].drop(columns=["customerID", "contract"]).to_string(index=False))
    print("─" * 72)
    print(f"Saved → {C.CUSTOMERS_CSV}")
    print(f"Saved → {C.TIMESERIES_CSV}")
    print(f"Saved → {C.TIMESERIES_SAMPLE_CSV}")


if __name__ == "__main__":
    customers, ts = generate()
    _print_preview(customers, ts)
