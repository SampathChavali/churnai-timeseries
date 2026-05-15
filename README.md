# ChurnAI · Time-Series

A complete, self-contained sub-system of the **ChurnAI** project: it focuses
exclusively on the **time-series** modality — 12 months of customer
behaviour fed into a BiLSTM with attention, served by a polished Streamlit
UI for documentation and demos.

```
Input  : (batch, 12 months, 6 features per month)
Model  : BiLSTM (2 layers, hidden=128, bidir.)  →  Attention  →  128-dim emb  →  sigmoid
Output : P(churn) per customer  +  per-month attention weights
```

No Kaggle download is required — a deterministic synthetic dataset is
generated locally on first run.

---

## Quick start

```bash
cd ~/Desktop/ChurnAI-TimeSeries
pip install -r requirements.txt

# 1. generate dataset, train LSTM, score every customer
python train.py

# 2. open the dashboard
streamlit run app/dashboard.py
```

`train.py` is idempotent — re-running it reuses the cached dataset and
just retrains. To force a fresh dataset, delete `data/*.csv`.

---

## Project layout

```
~/Desktop/ChurnAI-TimeSeries/
├── config.py                  All paths, hyperparameters, schema
├── generate_data.py           Synthetic time-series dataset generator
├── train.py                   End-to-end training + inference pipeline
├── requirements.txt
├── README.md
│
├── data/
│   ├── customers.csv          One row per customer  (2,000 rows)
│   ├── time_series.csv        12 months × customers (24,000 rows)
│   ├── time_series_sample.csv First 25 customers    (300 rows, for docs)
│   └── predictions.csv        Output of train.py
│
├── models/
│   ├── __init__.py
│   └── lstm_encoder.py        BiLSTM + attention model + training loop
│
├── app/
│   └── dashboard.py           5-page Streamlit UI
│
└── outputs/
    ├── saved/
    │   ├── lstm_encoder.pt    Trained PyTorch checkpoint
    │   └── ts_scaler.joblib   StandardScaler fit on train split
    └── metrics.json           Test metrics + per-epoch history
```

---

## The time-series dataset

The dataset is fully synthetic but **reproducible** — every customer's
sequence is seeded from a hash of their `customerID`, so re-running
`generate_data.py` always yields the same rows.

### `data/customers.csv` — 1 row per customer

| Column            | Type    | Example       | Notes                                          |
|-------------------|---------|---------------|------------------------------------------------|
| `customerID`      | string  | `0001-DSCQR`  | Telco-style ID, used as the join key           |
| `churn`           | int     | `0` / `1`     | Ground-truth label                             |
| `contract`        | string  | `Month-to-month` / `One year` / `Two year` | More month-to-month among churners |
| `internetService` | string  | `DSL` / `Fiber optic` / `No`               | Fiber optic over-represented among churners |
| `tenure_months`   | int     | `34`          | Total months as a customer                     |

#### About the customer ID format

IDs follow the Kaggle Telco Customer Churn convention:

- **Format**: `NNNN-AAAAA` — a 4-digit sequence number, a dash, and 5 random
  uppercase letters (e.g. `0982-QHUUQ`).
- **Synthetic but deterministic**: each ID is hashed (MD5) to seed a
  per-customer random generator. Re-running `python generate_data.py`
  always produces the same rows on any machine, so the dataset is fully
  reproducible.
- **No PII**: these IDs are stand-ins for whatever internal account number
  a real telco would use. They are not derived from any real customer.

**Talking point for class:**
> "Each customer in our dataset has a Kaggle-Telco-style ID like
> `0982-QHUUQ`. The full 2,000-customer base is synthetic but
> deterministic — every ID is hashed to seed its own random generator, so
> the same ID always maps to the same 12 monthly snapshots and the same
> churn label, regardless of who runs the code or when. That makes our
> results 100% reproducible for grading."

Overall churn rate ≈ **26 %** (matches the Kaggle Telco baseline).

### `data/time_series.csv` — 12 rows per customer

| Column             | Type   | Range / meaning                              |
|--------------------|--------|----------------------------------------------|
| `customerID`       | string | Foreign key → `customers.csv`                |
| `month`            | int    | 1 … 12 (1 = oldest, 12 = most recent)        |
| `data_gb`          | float  | GB used that month                           |
| `call_minutes`     | float  | Voice minutes used that month                |
| `monthly_charge`   | float  | Dollars billed that month                    |
| `support_tickets`  | int    | Tickets opened that month                    |
| `login_count`      | int    | App / portal logins that month               |
| `service_outages`  | int    | Outages experienced that month               |
| `churn_label`      | int    | 0 or 1 — copied from `customers.csv`         |
| `contract`         | string | Copied from `customers.csv` for convenience  |

#### Generation rules (summary)

| Behaviour          | Retained (Churn=0)            | Churner (Churn=1)                                          |
|--------------------|-------------------------------|-------------------------------------------------------------|
| Data usage         | Stable or +5 % drift           | Months 7–10 decline 15–30 %, months 11–12 drop 45–65 %      |
| Call minutes       | Stable                         | Tracks the same decline curve as data                       |
| Logins             | Stable, growing slightly       | Sharp drop in last 3 months                                 |
| Support tickets    | 0–1 per month                  | 0–1 baseline, **2–4 in last 3 months**                      |
| Service outages    | Mostly 0                       | 1–3 per month towards the end                                |
| Monthly charge     | Flat                           | **Drifts up ~10 %** — a frustration driver                  |

The full rules live in `generate_data.py::_generate_sequence`.

### `data/time_series_sample.csv` — for documentation

A 300-row slice (first 25 customers × 12 months) is written separately so
you can paste it straight into reports without dealing with the 24 k-row
full table.

---

## Streamlit UI — what each page does

| # | Page                     | Highlights                                                                                                                                  |
|---|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | **Risk Overview**        | 4 KPIs (total, high, medium, revenue at risk), donut + histogram, filterable table, CSV export                                              |
| 2 | **Customer Deep Dive**   | Pick any customer → 12-month trend chart, **live attention weights** from a real forward pass, last-3-vs-first-6 month delta table          |
| 3 | **Predict New Customer** | Two preset templates + sliders for months 10–12 → live LSTM forward pass → probability gauge, attention chart                               |
| 4 | **Cohort & Trends**      | Average behaviour by predicted tier, feature-of-the-day selector, stacked-bar risk mix by contract type                                     |
| 5 | **Model Performance**    | Test AUC / acc / recall / precision, train + val loss curves, val AUC curve, architecture table                                             |

UI style: glassmorphism over a dark `#030712` background with radial
purple / pink / cyan gradient accents, gradient text headings, and
gradient buttons — matches the larger ChurnAI brand.

---

## Model details

### Architecture (`models/lstm_encoder.py::LSTMEncoder`)

```
x  ∈  (batch, 12, 6)
   │
   ├── BiLSTM (input=6, hidden=128, layers=2, bidirectional, dropout=0.3)
   │        h  ∈  (batch, 12, 256)
   │
   ├── Additive attention
   │        scores  =  v · tanh(W h)            ∈  (batch, 12)
   │        weights =  softmax(scores)          ∈  (batch, 12)
   │        context =  Σ_t weight_t * h_t       ∈  (batch, 256)
   │
   ├── Projection
   │        Linear(256 → 128)  +  LayerNorm  +  Dropout
   │        embedding  ∈  (batch, 128)
   │
   └── Classifier
            Linear(128 → 1)  →  sigmoid
            P(churn)   ∈  (batch,)
```

### Training

- Loss      : `BCEWithLogitsLoss(pos_weight=2.5)` to balance the ~26 % positive class
- Optimizer : AdamW, lr=1e-3, weight_decay=1e-5
- Scheduler : `ReduceLROnPlateau(factor=0.5, patience=2)` on val loss
- Splits    : 70 % train / 15 % val / 15 % test (random, seeded)
- Epochs    : 15 (overridable in `config.py`)

### What gets saved

- `outputs/saved/lstm_encoder.pt` — best checkpoint (lowest val loss)
- `outputs/saved/ts_scaler.joblib` — `StandardScaler` fit on the training split only
- `outputs/metrics.json` — test metrics + per-epoch history for the UI

---

## How this fits into the bigger ChurnAI project

The larger ChurnAI multimodal system fuses three modalities:

```
LSTM (this repo) ─┐
DistilBERT       ─┼─→  Cross-Attention Fusion  →  Classifier  →  P(churn)
MLP tabular      ─┘
```

This sub-project is the time-series modality in isolation — useful for
documentation, ablation studies, and as a standalone "leading indicator"
service. Its API (`build_sequences`, `predict_with_attention`,
`load_model`) is what the multimodal pipeline imports.

---

## Where data enters the system

There are three entry points for data — useful to clarify when someone
asks "where is the live data?"

| Entry point | What it does | Trigger |
|---|---|---|
| **Batch scoring** | `train.py` reads `data/time_series.csv`, trains the BiLSTM (if no checkpoint exists), and writes `data/predictions.csv`. **The UI reads from this file** — it is not live-streaming. | `python train.py` |
| **Single-customer live scoring** | The *PREDICT* page lets you enter 12 monthly snapshots via sliders and feeds them straight into the loaded LSTM. The result is a real forward pass, computed live in the browser session. | Open the *PREDICT* page |
| **New historical data** | Append rows to `data/time_series.csv` (same schema as the generator) and re-run `train.py`. The model re-scores everyone, including the new customers. | Append CSV → `python train.py` |

The green pulse at the top of the dashboard means **the trained model is
loaded in memory and ready to score** — it does NOT mean data is
streaming in. There is no real-time feed in this demo, by design: the
goal of the project is to demonstrate the modeling approach, not to
build an ingestion pipeline.

---

## Reproducibility

All randomness is controlled:

- Dataset       : per-customer seeds derived from a hash of `customerID`
- Train / val / test split : seeded with `RANDOM_SEED = 42`
- PyTorch       : `torch.manual_seed(RANDOM_SEED)` in `train_lstm`

Two runs on the same machine produce **identical** dataset CSVs and
near-identical metrics (small variation from cuDNN/MPS non-determinism).
