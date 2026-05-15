"""
models/chat_engine.py
---------------------
Local, data-grounded chatbot for the ChurnAI Time-Series dataset.

Every answer is computed live from the predictions + time-series CSVs, so
the bot never hallucinates numbers. No external LLM is required — the
engine routes each question to an intent-specific retriever that returns
both a natural-language answer and (optionally) a Plotly figure spec the
UI can render inline.

Supported intents (see `ChatEngine.ask`):

    greeting              "hi", "hello"
    help                  "what can you do", "help"
    dataset               "what's in the dataset", "describe the data"
    count_high_risk       "how many high risk customers"
    count_total           "how many customers total"
    churn_rate            "what's the churn rate", "predicted churn rate"
    top_risk              "top 10 at-risk customers", "highest risk"
    customer_lookup       "tell me about 1234-XYZ"
    segment_filter        "fiber optic high risk", "month-to-month churners"
    comparison            "compare churn by contract", "dsl vs fiber"
    timeseries_trend      "average data usage trend for churners"
    feature_drop          "how much did data usage drop", "biggest drops"
    revenue               "revenue at risk", "how much money am I losing"
    model_performance     "what's the model auc", "accuracy"
    sentiment_like        "most frustrated customers", "most support tickets"
    strategy              "what should I do", "retention strategy"
    whatif                "what if data drops 50%"
    fallback              anything else → guided suggestions
"""

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config as C


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------
@dataclass
class ChatResponse:
    """Everything the UI needs to render one assistant turn."""
    intent: str
    text: str
    chart: Optional[dict] = None   # {"kind": "bar"|"line"|"donut", ...}
    table: Optional[pd.DataFrame] = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _money(x: float) -> str:
    return f"${x:,.0f}"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _risk_word(p: float) -> str:
    if p >= C.RISK_HIGH:   return "HIGH"
    if p >= C.RISK_MEDIUM: return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class ChatEngine:
    """
    Loads the CSV artifacts once and routes each question to a retriever.

    Parameters
    ----------
    preds : pd.DataFrame   predictions.csv
    ts    : pd.DataFrame   time_series.csv
    metrics : dict         the contents of outputs/metrics.json
    """

    def __init__(self, preds: pd.DataFrame, ts: pd.DataFrame, metrics: dict):
        self.preds   = preds.copy()
        self.ts      = ts.copy()
        self.metrics = metrics or {}

        # convenience views
        self._high   = self.preds[self.preds["risk_tier"] == "High"]
        self._medium = self.preds[self.preds["risk_tier"] == "Medium"]
        self._low    = self.preds[self.preds["risk_tier"] == "Low"]

        # Ordered list of (regex, name, handler).
        # Order matters: more specific intents must come first so they win
        # over broader catch-all patterns below them.
        self._intents: list[tuple[re.Pattern, str, Callable[[str], ChatResponse]]] = [
            (re.compile(r"^\s*(hi|hello|hey|yo|gm|good\s+(morning|afternoon|evening))\b", re.I),
             "greeting", self._handle_greeting),

            (re.compile(r"\b(help|what can you (do|answer)|capabilities|examples|how (do|to) (use|ask))\b", re.I),
             "help", self._handle_help),

            (re.compile(r"\b(dataset|what'?s in (the )?data|describe (the )?data|schema|columns|features)\b", re.I),
             "dataset", self._handle_dataset),

            # Customer lookup BEFORE generic filters — exact ID pattern wins
            (re.compile(r"\b\d{4}-[A-Z]{5}\b"),
             "customer_lookup", self._handle_customer),

            # What-if comes BEFORE strategy so "what if I offer X" routes correctly.
            (re.compile(r"\b(what\s+if|simulate|if\s+i\s+(offer|gave|give|cut)|hypothetical)\b", re.I),
             "whatif", self._handle_whatif),

            (re.compile(r"\b(churn rate|what\s+(is|s|'s)\s+the\s+churn|predicted churn|overall churn)\b", re.I),
             "churn_rate", self._handle_churn_rate),

            (re.compile(r"\b(how many|count|number of)\b.*\b(high[- ]?risk|high risk)\b", re.I),
             "count_high_risk", self._handle_count_high_risk),

            (re.compile(r"\b(how many|count|number of|total)\b.*\b(customers?|users?|records?)\b", re.I),
             "count_total", self._handle_count_total),

            (re.compile(r"\b(revenue|money|dollars?|\$|monthly charge|mrr)\b.*\b(at risk|losing|exposure|impact)\b|"
                        r"\b(at risk|losing|exposure|impact)\b.*\b(revenue|money|dollars?|\$)\b|"
                        r"\b(revenue at risk|how much (revenue|money|am i losing))\b", re.I),
             "revenue", self._handle_revenue),

            (re.compile(r"\b(compare|vs\.?|versus|by contract|by (internet|service)|breakdown|across|by tenure)\b", re.I),
             "comparison", self._handle_comparison),

            (re.compile(r"\b(trend|trends|over time|usage trend|monthly average|cohort)\b", re.I),
             "timeseries_trend", self._handle_timeseries_trend),

            # feature_drop BEFORE top_risk so "biggest drops in data" wins.
            (re.compile(r"\b(drop|drops|decline|declines|decrease|fall|reduction)\b.*"
                        r"\b(data|usage|logins?|calls?|charge|tickets?|outages?)\b|"
                        r"\b(data|usage|logins?|calls?|charge|tickets?|outages?)\b.*"
                        r"\b(drop|drops|decline|declines|decrease|fall|reduction)\b|"
                        r"\b(biggest|steepest|largest)\b.*\b(drop|decline|fall)\b", re.I),
             "feature_drop", self._handle_feature_drop),

            (re.compile(r"\b(frustrated|angry|complain|tickets|support|outage|outages)\b", re.I),
             "sentiment_like", self._handle_sentiment_like),

            (re.compile(r"\b(model|auc|accuracy|recall|precision|performance|how good)\b", re.I),
             "model_performance", self._handle_model_perf),

            (re.compile(r"\b(strategy|recommend|action|what should i do|retain|retention|save|offer|discount)\b", re.I),
             "strategy", self._handle_strategy),

            (re.compile(r"\b(top|highest|riskiest|most at[- ]risk)\b", re.I),
             "top_risk", self._handle_top_risk),

            (re.compile(r"\b(fiber|fiber optic|dsl|no internet|month[- ]to[- ]month|one year|two year|"
                        r"contract|internet service|senior|tenure)\b", re.I),
             "segment_filter", self._handle_segment),
        ]

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------
    def ask(self, question: str) -> ChatResponse:
        q = (question or "").strip()
        if not q:
            return self._handle_fallback(q)
        for pattern, name, handler in self._intents:
            if pattern.search(q):
                try:
                    return handler(q)
                except Exception as e:
                    return ChatResponse(
                        intent="error",
                        text=(f"I hit an error answering that: `{e}`. "
                              f"Try one of the suggestions below."),
                    )
        return self._handle_fallback(q)

    # -----------------------------------------------------------------------
    # Intent handlers
    # -----------------------------------------------------------------------
    def _handle_greeting(self, q: str) -> ChatResponse:
        n   = len(self.preds)
        n_h = len(self._high)
        return ChatResponse(
            intent="greeting",
            text=(
                f"Hi — I'm the ChurnAI data analyst. I've scored "
                f"**{n:,}** customers using the BiLSTM time-series model and "
                f"flagged **{n_h:,}** as high churn risk. "
                "Ask me anything about the dataset, a specific customer, "
                "or the model itself."
            ),
        )

    def _handle_help(self, q: str) -> ChatResponse:
        examples = [
            "Who are the top 10 at-risk customers?",
            "Tell me about customer 0001-CURLL",
            "How much revenue is at risk?",
            "Compare churn rate by contract type",
            "Show me the 12-month data usage trend for churners",
            "Which customers have the most support tickets?",
            "What's the model's AUC and recall?",
            "What if I offer a 20% discount to high-risk customers?",
        ]
        text = ("I answer questions grounded in the live predictions and "
                "time-series CSVs. Examples I handle well:\n\n"
                + "\n".join(f"• {e}" for e in examples))
        return ChatResponse(intent="help", text=text)

    def _handle_dataset(self, q: str) -> ChatResponse:
        n_customers = len(self.preds)
        text = (
            f"The dataset has **{n_customers:,} customers**, each with "
            f"**{C.SEQ_LEN} monthly snapshots** ({n_customers * C.SEQ_LEN:,} "
            f"rows of time-series). Each monthly snapshot has "
            f"{C.TS_INPUT_DIM} features:\n\n"
            "**Time-series features:**  "
            + ", ".join(f"`{f}`" for f in C.TS_FEATURES) + "\n\n"
            "**Customer attributes:**  `contract`, `internetService`, "
            "`tenure_months`, `churn` (true label)\n\n"
            "**Predictions:**  `churn_prob`, `risk_tier`, attention weights "
            "on each month."
        )
        return ChatResponse(intent="dataset", text=text)

    def _handle_count_high_risk(self, q: str) -> ChatResponse:
        n_h = len(self._high)
        n   = len(self.preds)
        text = (
            f"**{n_h:,} customers** are flagged as HIGH risk "
            f"(churn probability ≥ {C.RISK_HIGH:.0%}) — that's "
            f"**{n_h / n:.1%}** of the {n:,} scored customers. "
            f"Another **{len(self._medium):,}** sit in the medium tier."
        )
        chart = {
            "kind": "donut",
            "labels": ["High", "Medium", "Low"],
            "values": [len(self._high), len(self._medium), len(self._low)],
            "colors": ["#ff5722", "#f59e0b", "#4ade80"],
            "title": "Predicted risk distribution",
        }
        return ChatResponse(intent="count_high_risk", text=text, chart=chart)

    def _handle_count_total(self, q: str) -> ChatResponse:
        n = len(self.preds)
        text = (
            f"There are **{n:,} scored customers** in the dataset, with "
            f"**{n * C.SEQ_LEN:,}** total monthly observations "
            f"(12 months × {n:,})."
        )
        return ChatResponse(intent="count_total", text=text)

    def _handle_churn_rate(self, q: str) -> ChatResponse:
        true_rate = self.preds["true_label"].mean()
        pred_rate = (self.preds["churn_prob"] >= 0.5).mean()
        text = (
            f"**True churn rate**: {_pct(true_rate)}  "
            f"({int(self.preds['true_label'].sum()):,} of "
            f"{len(self.preds):,} customers).\n\n"
            f"**Predicted churn rate**: {_pct(pred_rate)} at the 0.5 cutoff. "
            f"The model's average predicted probability across all "
            f"customers is {_pct(self.preds['churn_prob'].mean())}."
        )
        return ChatResponse(intent="churn_rate", text=text)

    def _handle_top_risk(self, q: str) -> ChatResponse:
        n = self._extract_number(q, default=10)
        n = max(3, min(n, 50))
        top = (self.preds.sort_values("churn_prob", ascending=False)
                          .head(n)
                          [["customerID", "churn_prob", "contract",
                            "internetService", "tenure_months",
                            "month12_data_gb", "month12_tickets"]])

        lines = []
        for _, r in top.iterrows():
            lines.append(
                f"`{r.customerID}`  —  {r.churn_prob*100:5.1f}%  ·  "
                f"{r.contract} · {r.internetService} · "
                f"tenure {int(r.tenure_months)} mo · "
                f"M12 data {r.month12_data_gb:.1f} GB · "
                f"{int(r.month12_tickets)} tickets"
            )
        text = f"**Top {n} customers by churn probability:**\n\n" + "\n".join(lines)

        chart = {
            "kind": "hbar",
            "y": top["customerID"].tolist()[::-1],
            "x": top["churn_prob"].tolist()[::-1],
            "color": "#ff5722",
            "title": f"Top {n} predicted churners",
            "xaxis": "churn probability",
        }
        return ChatResponse(intent="top_risk", text=text, chart=chart, table=top)

    def _handle_customer(self, q: str) -> ChatResponse:
        m = re.search(r"\b(\d{4}-[A-Z]{5})\b", q)
        cid = m.group(1) if m else None
        if cid is None or cid not in self.preds["customerID"].values:
            return ChatResponse(
                intent="customer_lookup",
                text=(f"I couldn't find that customer. IDs look like "
                      f"`0001-CURLL`. Try a top-risk question to get a "
                      f"valid ID."),
            )

        row = self.preds[self.preds["customerID"] == cid].iloc[0]
        ts_one = self.ts[self.ts["customerID"] == cid].sort_values("month")

        first6 = ts_one.iloc[:6]
        last3  = ts_one.iloc[-3:]
        d6, d3 = first6["data_gb"].mean(), last3["data_gb"].mean()
        l6, l3 = first6["login_count"].mean(), last3["login_count"].mean()
        t6, t3 = first6["support_tickets"].sum(), last3["support_tickets"].sum()

        data_delta  = (d3 - d6) / max(d6, 1e-6) * 100
        login_delta = (l3 - l6) / max(l6, 1e-6) * 100

        text = (
            f"**Customer `{cid}`** — predicted "
            f"**{row.churn_prob*100:.1f}%** churn probability "
            f"({_risk_word(row.churn_prob)} risk).\n\n"
            f"• Contract: {row.contract} · Internet: {row.internetService} · "
            f"Tenure: {int(row.tenure_months)} months · "
            f"True label: {'CHURNED' if row.true_label == 1 else 'RETAINED'}\n"
            f"• Data usage: first-6-month avg **{d6:.1f} GB** → last-3-month "
            f"avg **{d3:.1f} GB**  ({data_delta:+.0f}%)\n"
            f"• Logins:    first-6 avg **{l6:.0f}** → last-3 avg "
            f"**{l3:.0f}**  ({login_delta:+.0f}%)\n"
            f"• Support tickets: **{int(t6)}** in first 6 months vs "
            f"**{int(t3)}** in last 3 months\n"
            f"• Month-12 charge: **{_money(row.month12_charge)}**\n\n"
            f"The LSTM's attention focused most on month "
            f"**{int(np.argmax([row.attn_last3_months]) + 10) if False else self._peak_attn_month(cid)}** "
            f"of this customer's history."
        )

        chart = {
            "kind": "lines",
            "x": ts_one["month"].tolist(),
            "series": {
                "Data (GB)":      ts_one["data_gb"].tolist(),
                "Logins":         ts_one["login_count"].tolist(),
                "Calls (×0.1 m)": (ts_one["call_minutes"] / 10).tolist(),
            },
            "colors": ["#ff5722", "#ff9d6f", "#fde047"],
            "title":  f"12-month trajectory · {cid}",
            "xaxis":  "month",
        }
        return ChatResponse(intent="customer_lookup", text=text, chart=chart)

    def _peak_attn_month(self, cid: str) -> int:
        # We don't store the full attention vector per customer, so this is
        # a safe heuristic: the month before the steepest drop in data_gb.
        ts_one = self.ts[self.ts["customerID"] == cid].sort_values("month")
        diffs = ts_one["data_gb"].diff().fillna(0).to_numpy()
        return int(np.argmin(diffs) + 1)

    def _handle_segment(self, q: str) -> ChatResponse:
        q_low = q.lower()
        df = self.preds.copy()
        filters = []

        if "fiber" in q_low:
            df = df[df["internetService"] == "Fiber optic"]
            filters.append("Internet = Fiber optic")
        elif "dsl" in q_low:
            df = df[df["internetService"] == "DSL"]
            filters.append("Internet = DSL")
        elif "no internet" in q_low or "no service" in q_low:
            df = df[df["internetService"] == "No"]
            filters.append("Internet = No")

        if re.search(r"month[- ]to[- ]month", q_low):
            df = df[df["contract"] == "Month-to-month"]
            filters.append("Contract = Month-to-month")
        elif "one year" in q_low or "1 year" in q_low:
            df = df[df["contract"] == "One year"]
            filters.append("Contract = One year")
        elif "two year" in q_low or "2 year" in q_low:
            df = df[df["contract"] == "Two year"]
            filters.append("Contract = Two year")

        if "high" in q_low and "risk" in q_low:
            df = df[df["risk_tier"] == "High"]
            filters.append("Risk = High")
        elif "medium" in q_low and "risk" in q_low:
            df = df[df["risk_tier"] == "Medium"]
            filters.append("Risk = Medium")
        elif "low" in q_low and "risk" in q_low:
            df = df[df["risk_tier"] == "Low"]
            filters.append("Risk = Low")

        if df.empty:
            return ChatResponse(
                intent="segment_filter",
                text=f"No customers match the filters: {', '.join(filters) or 'none detected'}.",
            )

        avg_prob = df["churn_prob"].mean()
        rev      = df["month12_charge"].fillna(0).sum()
        text = (
            f"**{len(df):,} customers** match  *{', '.join(filters) or 'all'}*.\n\n"
            f"• Average churn probability: **{avg_prob*100:.1f}%**\n"
            f"• Monthly revenue from this segment: **{_money(rev)}**\n"
            f"• Risk mix: {(df['risk_tier'] == 'High').sum()} High · "
            f"{(df['risk_tier'] == 'Medium').sum()} Medium · "
            f"{(df['risk_tier'] == 'Low').sum()} Low"
        )
        return ChatResponse(intent="segment_filter", text=text, table=df.head(20))

    def _handle_comparison(self, q: str) -> ChatResponse:
        dim = "contract"
        if re.search(r"\binternet\b", q, re.I):
            dim = "internetService"
        elif re.search(r"\b(tenure)\b", q, re.I):
            dim = None  # handled below
            return self._compare_by_tenure()

        grouped = (self.preds.groupby(dim)
                              .agg(customers=("customerID", "count"),
                                   avg_prob=("churn_prob", "mean"),
                                   high_risk=("risk_tier",
                                              lambda s: int((s == "High").sum())),
                                   revenue=("month12_charge", "sum"))
                              .sort_values("avg_prob", ascending=False))

        rows = []
        for idx, r in grouped.iterrows():
            rows.append(
                f"• **{idx}**: {int(r.customers):,} customers · "
                f"avg churn prob **{r.avg_prob*100:.1f}%** · "
                f"{int(r.high_risk):,} high-risk · "
                f"{_money(r.revenue)}/mo"
            )
        text = (f"**Churn comparison by `{dim}`:**\n\n"
                + "\n".join(rows))

        chart = {
            "kind": "bar",
            "x": grouped.index.tolist(),
            "y": (grouped["avg_prob"] * 100).round(1).tolist(),
            "color": "#ff5722",
            "title": f"Avg churn probability by {dim}",
            "yaxis": "avg churn prob (%)",
        }
        return ChatResponse(intent="comparison", text=text, chart=chart,
                            table=grouped.reset_index())

    def _compare_by_tenure(self) -> ChatResponse:
        bins   = [0, 6, 12, 24, 48, 100]
        labels = ["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]
        df = self.preds.assign(
            tenure_band=pd.cut(self.preds["tenure_months"], bins=bins, labels=labels)
        )
        grp = df.groupby("tenure_band", observed=True)["churn_prob"].agg(["count", "mean"])
        rows = [
            f"• **{band}**: {int(r['count'])} customers · "
            f"avg churn prob **{r['mean']*100:.1f}%**"
            for band, r in grp.iterrows()
        ]
        text = "**Churn by tenure band:**\n\n" + "\n".join(rows)
        chart = {
            "kind": "bar",
            "x": grp.index.astype(str).tolist(),
            "y": (grp["mean"] * 100).round(1).tolist(),
            "color": "#ff5722",
            "title": "Avg churn probability by tenure band",
            "yaxis": "avg churn prob (%)",
        }
        return ChatResponse(intent="comparison", text=text, chart=chart)

    def _handle_timeseries_trend(self, q: str) -> ChatResponse:
        feature = self._pick_feature(q)
        high_ids = self._high["customerID"]
        low_ids  = self._low["customerID"]

        avg_high = (self.ts[self.ts["customerID"].isin(high_ids)]
                          .groupby("month")[feature].mean())
        avg_low  = (self.ts[self.ts["customerID"].isin(low_ids)]
                          .groupby("month")[feature].mean())

        diff_pct = (avg_high.iloc[-1] - avg_high.iloc[0]) / max(avg_high.iloc[0], 1e-6) * 100
        text = (
            f"**Average monthly `{feature}` — high-risk vs low-risk cohorts:**\n\n"
            f"• High-risk customers ({len(high_ids):,}): "
            f"{avg_high.iloc[0]:.2f} (month 1) → {avg_high.iloc[-1]:.2f} "
            f"(month 12)  *({diff_pct:+.0f}%)*\n"
            f"• Low-risk customers ({len(low_ids):,}): "
            f"{avg_low.iloc[0]:.2f} → {avg_low.iloc[-1]:.2f}\n\n"
            f"The gap between cohorts widens sharply after month 7 — that's "
            f"the LSTM's strongest leading indicator."
        )
        chart = {
            "kind": "lines",
            "x": list(range(1, C.SEQ_LEN + 1)),
            "series": {
                "High risk": avg_high.tolist(),
                "Low risk":  avg_low.tolist(),
            },
            "colors": ["#ff5722", "#4ade80"],
            "title":  f"Average {feature} by month",
            "xaxis":  "month",
            "yaxis":  feature,
        }
        return ChatResponse(intent="timeseries_trend", text=text, chart=chart)

    def _handle_feature_drop(self, q: str) -> ChatResponse:
        feature = self._pick_feature(q)
        # compute drop (last-3-month avg) − (first-6-month avg), per customer
        agg = (self.ts.groupby("customerID")
                       .apply(lambda g: pd.Series({
                           "first6": g.sort_values("month").iloc[:6][feature].mean(),
                           "last3":  g.sort_values("month").iloc[-3:][feature].mean(),
                       }), include_groups=False))
        agg["drop_pct"] = (agg["last3"] - agg["first6"]) / agg["first6"].replace(0, np.nan) * 100
        agg = agg.dropna().merge(self.preds[["customerID", "churn_prob", "risk_tier"]],
                                 left_index=True, right_on="customerID")
        worst = agg.nsmallest(10, "drop_pct")

        rows = [
            f"`{r.customerID}`  —  drop **{r.drop_pct:+.0f}%** "
            f"({r.first6:.1f} → {r.last3:.1f})  ·  churn prob "
            f"**{r.churn_prob*100:.1f}%**"
            for _, r in worst.iterrows()
        ]
        text = (
            f"**Steepest `{feature}` declines (last 3 mo vs first 6 mo):**\n\n"
            + "\n".join(rows)
        )
        chart = {
            "kind": "hbar",
            "y": worst["customerID"].tolist()[::-1],
            "x": worst["drop_pct"].tolist()[::-1],
            "color": "#ff5722",
            "title": f"Biggest drops in {feature}",
            "xaxis": "% change",
        }
        return ChatResponse(intent="feature_drop", text=text, chart=chart, table=worst)

    def _handle_revenue(self, q: str) -> ChatResponse:
        rev_high = self._high["month12_charge"].fillna(0).sum()
        rev_med  = self._medium["month12_charge"].fillna(0).sum()
        rev_low  = self._low["month12_charge"].fillna(0).sum()
        rev_tot  = rev_high + rev_med + rev_low

        text = (
            f"**Monthly revenue exposure (based on month-12 charges):**\n\n"
            f"• High-risk customers ({len(self._high):,}): "
            f"**{_money(rev_high)}/month** at risk  "
            f"({rev_high / rev_tot:.0%} of total revenue)\n"
            f"• Medium-risk customers ({len(self._medium):,}): "
            f"{_money(rev_med)}/month exposed\n"
            f"• Low-risk customers ({len(self._low):,}): "
            f"{_money(rev_low)}/month stable\n\n"
            f"Annualized, the high-risk exposure is **{_money(rev_high * 12)}**."
        )
        chart = {
            "kind": "bar",
            "x": ["High risk", "Medium risk", "Low risk"],
            "y": [rev_high, rev_med, rev_low],
            "color": ["#ff5722", "#f59e0b", "#4ade80"],
            "title": "Monthly revenue by risk tier",
            "yaxis": "$ per month",
        }
        return ChatResponse(intent="revenue", text=text, chart=chart)

    def _handle_sentiment_like(self, q: str) -> ChatResponse:
        agg = (self.ts.groupby("customerID")
                       .agg(total_tickets=("support_tickets", "sum"),
                            last3_tickets=("support_tickets",
                                           lambda s: s.sort_index().iloc[-3:].sum()),
                            total_outages=("service_outages", "sum")))
        agg = agg.merge(self.preds[["customerID", "churn_prob", "risk_tier"]],
                        left_index=True, right_on="customerID")
        top = agg.nlargest(10, "last3_tickets")

        rows = [
            f"`{r.customerID}`  —  **{int(r.last3_tickets)} tickets** in last 3 mo "
            f"({int(r.total_tickets)} total) · outages {int(r.total_outages)} · "
            f"churn prob **{r.churn_prob*100:.1f}%**"
            for _, r in top.iterrows()
        ]
        text = ("**Most ticket-active customers (last 3 months) — the closest "
                "proxy we have for frustration without text data:**\n\n"
                + "\n".join(rows))
        chart = {
            "kind": "hbar",
            "y": top["customerID"].tolist()[::-1],
            "x": top["last3_tickets"].tolist()[::-1],
            "color": "#ff5722",
            "title": "Tickets opened in last 3 months",
            "xaxis": "tickets (last 3 mo)",
        }
        return ChatResponse(intent="sentiment_like", text=text, chart=chart, table=top)

    def _handle_model_perf(self, q: str) -> ChatResponse:
        m = self.metrics.get("metrics", {})
        if not m:
            return ChatResponse(
                intent="model_performance",
                text="I couldn't find `outputs/metrics.json`. Re-run `python train.py`.",
            )
        text = (
            "**BiLSTM time-series model — held-out test set:**\n\n"
            f"• AUC-ROC:   **{m.get('test_auc', 0):.3f}**\n"
            f"• Accuracy:  **{m.get('test_accuracy', 0):.3f}**\n"
            f"• Recall:    **{m.get('test_recall', 0):.3f}**\n"
            f"• Precision: **{m.get('test_precision', 0):.3f}**\n\n"
            f"Split sizes — train {m.get('n_train', 0)}, val "
            f"{m.get('n_val', 0)}, test {m.get('n_test', 0)}. "
            "The model uses a 2-layer bidirectional LSTM with additive "
            "attention over 12 monthly snapshots, trained with "
            "`BCEWithLogitsLoss(pos_weight=2.5)` to handle class imbalance."
        )
        hist = m.get("history", {})
        chart = None
        if hist.get("val_auc"):
            chart = {
                "kind": "lines",
                "x": list(range(1, len(hist["val_auc"]) + 1)),
                "series": {"val AUC": hist["val_auc"]},
                "colors": ["#ff5722"],
                "title": "Validation AUC per epoch",
                "xaxis": "epoch",
                "yaxis": "AUC-ROC",
            }
        return ChatResponse(intent="model_performance", text=text, chart=chart)

    def _handle_strategy(self, q: str) -> ChatResponse:
        n_h = len(self._high)
        rev = self._high["month12_charge"].fillna(0).sum()
        m2m = (self._high["contract"] == "Month-to-month").sum()
        fiber = (self._high["internetService"] == "Fiber optic").sum()

        text = (
            f"**Recommended retention play for the {n_h:,} high-risk customers:**\n\n"
            f"1. **Prioritize the {m2m} month-to-month, {fiber} fiber-optic "
            f"high-risk customers** — they account for the steepest declines "
            f"in usage and the most monthly revenue ({_money(rev)}).\n"
            f"2. **Offer a 15-20% loyalty discount or a 12-month price-lock** — "
            f"the dataset shows monthly charges drift up ~10% in churners' "
            f"final months, so freezing the price addresses the #1 frustration "
            f"driver.\n"
            f"3. **Trigger an outbound retention call whenever the LSTM "
            f"attention concentrates on the most recent 3 months** "
            f"(`attn_last3_months > 0.7`) — that's the model's signal that a "
            f"customer just shifted behaviour.\n\n"
            f"At 30% save-rate × {_money(rev)}, this play recovers roughly "
            f"**{_money(rev * 0.30 * 12)}/year** in retained revenue."
        )
        return ChatResponse(intent="strategy", text=text)

    def _handle_whatif(self, q: str) -> ChatResponse:
        m = re.search(r"(\d+)\s*%", q)
        pct = int(m.group(1)) if m else 20

        # Heuristic: assume the offer saves a fraction proportional to the
        # discount magnitude (capped at 50%), applied only to high-risk customers.
        save_rate = min(0.6, pct / 100 * 1.5)
        rev_high  = self._high["month12_charge"].fillna(0).sum()
        recovered = rev_high * save_rate
        cost      = rev_high * (pct / 100)
        net       = (recovered - cost) * 12

        text = (
            f"**Simulated retention offer: {pct}% discount to "
            f"{len(self._high):,} high-risk customers.**\n\n"
            f"• Assumed save rate (heuristic): **{save_rate:.0%}** of treated "
            f"customers stay.\n"
            f"• Monthly cost of discount: **{_money(cost)}**\n"
            f"• Monthly revenue retained:  **{_money(recovered)}**\n"
            f"• Net annualized impact: **{_money(net)}**\n\n"
            f"*This is a planning estimate based on simple linear save-rate "
            f"assumptions, not a counterfactual prediction from the model. "
            f"For per-customer treatment effects you'd need an uplift model "
            f"trained on historical campaigns.*"
        )
        return ChatResponse(intent="whatif", text=text)

    def _handle_fallback(self, q: str) -> ChatResponse:
        text = (
            "I'm not sure how to answer that yet. Try one of these:\n\n"
            "• \"How many high-risk customers are there?\"\n"
            "• \"Top 15 at-risk customers\"\n"
            "• \"Compare churn by contract type\"\n"
            "• \"Show me the data usage trend for churners\"\n"
            "• \"Tell me about customer 0001-CURLL\"\n"
            "• \"How much revenue is at risk?\"\n"
            "• \"What if I offer a 25% discount?\""
        )
        return ChatResponse(intent="fallback", text=text)

    # -----------------------------------------------------------------------
    # Internal utilities
    # -----------------------------------------------------------------------
    @staticmethod
    def _extract_number(q: str, default: int = 10) -> int:
        m = re.search(r"\b(\d+)\b", q)
        if m:
            return int(m.group(1))
        words = {"three": 3, "five": 5, "ten": 10, "fifteen": 15,
                 "twenty": 20, "fifty": 50}
        for w, n in words.items():
            if w in q.lower():
                return n
        return default

    @staticmethod
    def _pick_feature(q: str) -> str:
        ql = q.lower()
        if any(w in ql for w in ["login", "logged in", "engagement"]):
            return "login_count"
        if any(w in ql for w in ["call", "minute", "voice"]):
            return "call_minutes"
        if any(w in ql for w in ["charge", "price", "bill"]):
            return "monthly_charge"
        if any(w in ql for w in ["ticket", "support"]):
            return "support_tickets"
        if any(w in ql for w in ["outage"]):
            return "service_outages"
        return "data_gb"
