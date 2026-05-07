"""
DACARag - Business Intelligence Dashboard
==========================================

Trade-level KPIs and charts sourced live from the trades_quotes database.

Data-modelling note (for code readers, not shown in UI)
-------------------------------------------------------
Trade is treated as living on `job_types` only - the source of truth.
Although `invoices` carries a denormalised `trade` field, this dashboard
joins `invoices.job_type_id` -> `job_types.job_type_id` and reads the
trade from there. A trade reclassification on a job_type would be
reflected immediately across all historical invoices, with no risk of
stale denormalised values.

Schema (as exported from MongoDB Atlas, 2026-05-04):
    customers     (121 docs) - customer_id, first/last name, county, eircode...
    invoices      (150 docs) - invoice_id, customer_id, job_type_id,
                                invoice_date, total_inc_vat, ...
    invoice_items (750 docs) - invoice_id, item_id, item_name, quantity,
                                unit_price_ex_vat, line_total_ex_vat
    items          (88 docs) - item_id, item_name, category, unit_price_ex_vat
    job_types      (60 docs) - job_type_id, job_name, trade, ...
    pos            (15 docs) - po_number, po_date (string), job_type (string),
                                materials_subtotal, total_inc_vat, line_items[]

Foreign keys are STRING fields. All $lookup joins use string equality.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo import MongoClient

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

COLLECTIONS = {
    "customers": "customers",
    "invoices": "invoices",
    "invoice_items": "invoice_items",
    "purchase_orders": "pos",
    "items": "items",
    "job_types": "job_types",
}

# Autumn pastel palette
CARD_BG = "#FDFAF6"
TERRA = "#B85A5A"
SAGE = "#587858"
SAGE_PALE = "#E8EEE3"
BORDER = "#E5DDD2"
TEXT = "#2D2520"
MUTED = "#786558"

# Distinct accent colour per trade for the multi-trade revenue chart.
DEFAULT_TRADE_PALETTE = ["#A8B5C9", "#C9A57B", "#D4896B", "#A89484", "#8FA08F"]

THEME_CSS = f"""
<style>
.kpi-card {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    background: {CARD_BG};
    height: 100%;
}}
.kpi-title {{
    font-size: 0.82rem;
    color: {MUTED};
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}}
.kpi-value {{
    font-size: 1.6rem;
    font-weight: 800;
    color: {TEXT};
    margin: 6px 0 4px 0;
}}
.kpi-sub {{
    font-size: 0.78rem;
    color: {MUTED};
    margin: 0;
}}
.dashboard-title {{
    color: {TERRA};
    font-weight: 800;
    margin-bottom: 1rem;
}}
.section-header {{
    color: {TEXT};
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.6rem;
}}
hr.dashboard-rule {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 1.2rem 0;
}}
</style>
"""

# -----------------------------------------------------------------------------
# Connection
# -----------------------------------------------------------------------------


@st.cache_resource
def get_mongo_db():
    uri = os.getenv("MONGO_URI") or st.secrets.get("MONGO_URI", None)
    if not uri:
        st.error(
            "MONGO_URI not configured. Set it in `.env` or "
            ".streamlit/secrets.toml."
        )
        st.stop()
    db_name = os.getenv("MONGO_DB", "trades_quotes")
    client = MongoClient(uri)
    return client[db_name]


# -----------------------------------------------------------------------------
# Pipeline helpers
# -----------------------------------------------------------------------------


def _date_match(start: Optional[date], end: Optional[date], field: str) -> Optional[dict]:
    if start is None or end is None:
        return None
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())
    return {"$match": {field: {"$gte": start_dt, "$lte": end_dt}}}


def _trade_match(trade: str, field: str) -> Optional[dict]:
    if trade == "All":
        return None
    return {"$match": {field: trade}}


def _invoice_job_lookup(local_prefix: str = "") -> list:
    """
    Join an invoice document to its job_type. Trade lives on job_types,
    not on the invoice (we do not trust the denormalised invoice.trade).
    """
    local_field = f"{local_prefix}.job_type_id" if local_prefix else "job_type_id"
    return [
        {
            "$lookup": {
                "from": COLLECTIONS["job_types"],
                "localField": local_field,
                "foreignField": "job_type_id",
                "as": "job",
            }
        },
        {"$unwind": "$job"},
    ]


def _po_job_lookup() -> list:
    """
    Lookup pos.job_type against job_types.job_type_id OR job_types.job_name.
    The pos.job_type field may hold either form depending on PDF extraction.
    """
    return [
        {
            "$lookup": {
                "from": COLLECTIONS["job_types"],
                "let": {"j": "$job_type"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$or": [
                                    {"$eq": ["$job_type_id", "$$j"]},
                                    {"$eq": ["$job_name", "$$j"]},
                                ]
                            }
                        }
                    }
                ],
                "as": "job",
            }
        },
        {"$unwind": {"path": "$job", "preserveNullAndEmptyArrays": False}},
    ]


# -----------------------------------------------------------------------------
# Cached metadata queries
# -----------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def get_available_trades() -> list[str]:
    db = get_mongo_db()
    try:
        trades = db[COLLECTIONS["job_types"]].distinct("trade")
        return sorted([t for t in trades if t])
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def get_invoice_date_range() -> tuple[Optional[date], Optional[date]]:
    db = get_mongo_db()
    try:
        first = db[COLLECTIONS["invoices"]].find_one(
            {}, sort=[("invoice_date", 1)], projection={"invoice_date": 1}
        )
        last = db[COLLECTIONS["invoices"]].find_one(
            {}, sort=[("invoice_date", -1)], projection={"invoice_date": 1}
        )
        if first and last:
            return first["invoice_date"].date(), last["invoice_date"].date()
    except Exception:
        pass
    return None, None


# -----------------------------------------------------------------------------
# KPI queries
# -----------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def query_kpi_customer_count(trade: str, start: Optional[date], end: Optional[date]) -> int:
    db = get_mongo_db()
    pipeline: list = []
    date_stage = _date_match(start, end, "invoice_date")
    if date_stage:
        pipeline.append(date_stage)
    pipeline += _invoice_job_lookup()
    trade_stage = _trade_match(trade, "job.trade")
    if trade_stage:
        pipeline.append(trade_stage)
    pipeline += [
        {"$group": {"_id": "$customer_id"}},
        {"$count": "total"},
    ]
    result = list(db[COLLECTIONS["invoices"]].aggregate(pipeline))
    return result[0]["total"] if result else 0


@st.cache_data(ttl=300, show_spinner=False)
def query_kpi_jobs_value(trade: str, start: Optional[date], end: Optional[date]) -> float:
    db = get_mongo_db()
    pipeline: list = []
    date_stage = _date_match(start, end, "invoice_date")
    if date_stage:
        pipeline.append(date_stage)
    pipeline += _invoice_job_lookup()
    trade_stage = _trade_match(trade, "job.trade")
    if trade_stage:
        pipeline.append(trade_stage)
    pipeline += [{"$group": {"_id": None, "total": {"$sum": "$total_inc_vat"}}}]
    result = list(db[COLLECTIONS["invoices"]].aggregate(pipeline))
    return float(result[0]["total"]) if result else 0.0


@st.cache_data(ttl=300, show_spinner=False)
def query_kpi_materials_spend(trade: str, start: Optional[date], end: Optional[date]) -> float:
    db = get_mongo_db()
    pipeline: list = list(_po_job_lookup())
    trade_stage = _trade_match(trade, "job.trade")
    if trade_stage:
        pipeline.append(trade_stage)
    pipeline += [{"$group": {"_id": None, "total": {"$sum": "$materials_subtotal"}}}]
    result = list(db[COLLECTIONS["purchase_orders"]].aggregate(pipeline))
    return float(result[0]["total"]) if result else 0.0


# -----------------------------------------------------------------------------
# Chart queries
# -----------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def query_revenue_trend(trade: str, start: Optional[date], end: Optional[date]) -> pd.DataFrame:
    db = get_mongo_db()
    pipeline: list = []
    date_stage = _date_match(start, end, "invoice_date")
    if date_stage:
        pipeline.append(date_stage)
    pipeline += _invoice_job_lookup()
    trade_stage = _trade_match(trade, "job.trade")
    if trade_stage:
        pipeline.append(trade_stage)
    pipeline += [
        {
            "$group": {
                "_id": {
                    "month": {
                        "$dateToString": {"format": "%Y-%m", "date": "$invoice_date"}
                    },
                    "trade": "$job.trade",
                },
                "revenue": {"$sum": "$total_inc_vat"},
                "invoice_count": {"$sum": 1},
            }
        },
        {"$sort": {"_id.month": 1}},
    ]
    rows = list(db[COLLECTIONS["invoices"]].aggregate(pipeline))
    if not rows:
        return pd.DataFrame(columns=["month", "trade", "revenue", "invoice_count"])
    return pd.DataFrame(
        [
            {
                "month": r["_id"]["month"],
                "trade": r["_id"]["trade"],
                "revenue": float(r["revenue"]),
                "invoice_count": int(r["invoice_count"]),
            }
            for r in rows
        ]
    )


@st.cache_data(ttl=300, show_spinner=False)
def query_top_items(
    trade: str, start: Optional[date], end: Optional[date], limit: int = 10
) -> pd.DataFrame:
    db = get_mongo_db()
    pipeline: list = [
        {
            "$lookup": {
                "from": COLLECTIONS["invoices"],
                "localField": "invoice_id",
                "foreignField": "invoice_id",
                "as": "inv",
            }
        },
        {"$unwind": "$inv"},
    ]
    date_stage = _date_match(start, end, "inv.invoice_date")
    if date_stage:
        pipeline.append(date_stage)
    pipeline += [
        {
            "$lookup": {
                "from": COLLECTIONS["job_types"],
                "localField": "inv.job_type_id",
                "foreignField": "job_type_id",
                "as": "job",
            }
        },
        {"$unwind": "$job"},
    ]
    trade_stage = _trade_match(trade, "job.trade")
    if trade_stage:
        pipeline.append(trade_stage)
    pipeline += [
        {
            "$group": {
                "_id": "$item_name",
                "total_value": {"$sum": "$line_total_ex_vat"},
                "total_qty": {"$sum": "$quantity"},
            }
        },
        {"$sort": {"total_value": -1}},
        {"$limit": limit},
    ]
    rows = list(db[COLLECTIONS["invoice_items"]].aggregate(pipeline))
    if not rows:
        return pd.DataFrame(columns=["item", "total_value", "total_qty"])
    return pd.DataFrame(
        [
            {
                "item": r["_id"],
                "total_value": float(r["total_value"]),
                "total_qty": int(r["total_qty"]),
            }
            for r in rows
        ]
    )


@st.cache_data(ttl=300, show_spinner=False)
def query_county_activity(trade: str, start: Optional[date], end: Optional[date]) -> pd.DataFrame:
    db = get_mongo_db()
    pipeline: list = []
    date_stage = _date_match(start, end, "invoice_date")
    if date_stage:
        pipeline.append(date_stage)
    pipeline += _invoice_job_lookup()
    trade_stage = _trade_match(trade, "job.trade")
    if trade_stage:
        pipeline.append(trade_stage)
    pipeline += [
        {
            "$lookup": {
                "from": COLLECTIONS["customers"],
                "localField": "customer_id",
                "foreignField": "customer_id",
                "as": "customer",
            }
        },
        {"$unwind": "$customer"},
        {
            "$group": {
                "_id": "$customer.county",
                "invoice_count": {"$sum": 1},
                "revenue": {"$sum": "$total_inc_vat"},
            }
        },
        {"$sort": {"revenue": -1}},
    ]
    rows = list(db[COLLECTIONS["invoices"]].aggregate(pipeline))
    if not rows:
        return pd.DataFrame(columns=["county", "invoice_count", "revenue"])
    return pd.DataFrame(
        [
            {
                "county": r["_id"] or "Unknown",
                "invoice_count": int(r["invoice_count"]),
                "revenue": float(r["revenue"]),
            }
            for r in rows
        ]
    )


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------


def _kpi_card(title: str, value: str, sub: Optional[str] = None) -> str:
    sub_html = f'<p class="kpi-sub">{sub}</p>' if sub else ""
    return (
        f'<div class="kpi-card">'
        f'<p class="kpi-title">{title}</p>'
        f'<p class="kpi-value">{value}</p>'
        f"{sub_html}"
        f"</div>"
    )


def _fmt_money(x: float) -> str:
    return f"€{x:,.0f}"


def _styled_plotly_layout(fig):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color=TEXT, size=12),
        margin=dict(t=10, b=40, l=20, r=20),
    )
    return fig


def _trade_palette(trades: list[str]) -> dict:
    return {
        t: DEFAULT_TRADE_PALETTE[i % len(DEFAULT_TRADE_PALETTE)]
        for i, t in enumerate(sorted(trades))
    }


# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------


def _render_sidebar_filters(available_trades: list[str]) -> tuple:
    with st.sidebar:
        st.markdown("### Dashboard filters")

        trade = st.radio(
            "Trade",
            ["All"] + available_trades,
            index=0,
            key="bi_trade",
        )

        all_time = st.checkbox(
            "All time",
            value=True,
            key="bi_all_time",
            help="Show every invoice, no date filter.",
        )

        if all_time:
            start, end = None, None
            st.caption("Showing all dates")
        else:
            data_start, data_end = get_invoice_date_range()
            default_start = data_start or date.today() - timedelta(days=365)
            default_end = data_end or date.today()
            date_range = st.date_input(
                "Date range",
                value=(default_start, default_end),
                key="bi_date_range",
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = date_range
            else:
                start, end = None, None

        if st.button("Refresh data", use_container_width=True, key="bi_refresh"):
            st.cache_data.clear()
            st.rerun()

        st.divider()

    return trade, start, end


# -----------------------------------------------------------------------------
# Main render function
# -----------------------------------------------------------------------------


def render_dashboard() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    st.markdown(
        '<h1 class="dashboard-title">Business Intelligence</h1>',
        unsafe_allow_html=True,
    )

    available_trades = get_available_trades()
    if not available_trades:
        st.warning(
            "Could not read trade values from `job_types.trade`. "
            "Check your MongoDB connection in `.env`."
        )
        return

    trade, start, end = _render_sidebar_filters(available_trades)
    palette = _trade_palette(available_trades)

    # KPI row
    try:
        cust_count = query_kpi_customer_count(trade, start, end)
        jobs_value = query_kpi_jobs_value(trade, start, end)
        po_spend = query_kpi_materials_spend(trade, start, end)
    except Exception as e:
        st.error(f"KPI query failed: {e}")
        return

    period_label = (
        f"{start.strftime('%b %Y')} – {end.strftime('%b %Y')}"
        if start and end
        else "all time"
    )
    sub_label = "all trades" if trade == "All" else trade.lower()

    kc1, kc2, kc3 = st.columns(3, gap="medium")
    with kc1:
        st.markdown(
            _kpi_card("Customers served", f"{cust_count}", f"{sub_label} · {period_label}"),
            unsafe_allow_html=True,
        )
    with kc2:
        st.markdown(
            _kpi_card("Total jobs value", _fmt_money(jobs_value), f"{sub_label} · inc VAT"),
            unsafe_allow_html=True,
        )
    with kc3:
        st.markdown(
            _kpi_card(
                "Materials spend (POs)",
                _fmt_money(po_spend),
                f"{sub_label} · all POs (15 total)",
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="dashboard-rule">', unsafe_allow_html=True)

    # Chart 1: Revenue trend
    st.markdown('<h3 class="section-header">Revenue trend</h3>', unsafe_allow_html=True)
    df_rev = query_revenue_trend(trade, start, end)
    if df_rev.empty:
        st.info("No invoices in this period for the selected trade.")
    else:
        fig = px.line(
            df_rev,
            x="month",
            y="revenue",
            color="trade",
            color_discrete_map=palette,
            markers=True,
            labels={"month": "Month", "revenue": "Revenue (€)", "trade": "Trade"},
        )
        _styled_plotly_layout(fig)
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#eee", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    # Chart 2: Top items
    st.markdown(
        '<h3 class="section-header">Top 10 catalogue items by invoice value</h3>',
        unsafe_allow_html=True,
    )
    df_items = query_top_items(trade, start, end, limit=10)
    if df_items.empty:
        st.info("No line items in this period.")
    else:
        df_items = df_items.sort_values("total_value", ascending=True)
        fig = px.bar(
            df_items,
            x="total_value",
            y="item",
            orientation="h",
            color_discrete_sequence=[TERRA],
            labels={"total_value": "Total value (€)", "item": ""},
        )
        _styled_plotly_layout(fig)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(showgrid=True, gridcolor="#eee", tickformat=",.0f")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    # Chart 3: County activity (sage gradient)
    st.markdown(
        '<h3 class="section-header">Activity by county</h3>',
        unsafe_allow_html=True,
    )
    df_county = query_county_activity(trade, start, end)
    if df_county.empty:
        st.info("No customer activity in this period.")
    else:
        df_county = df_county.head(15).sort_values("revenue", ascending=True)
        fig = px.bar(
            df_county,
            x="revenue",
            y="county",
            orientation="h",
            color="invoice_count",
            color_continuous_scale=[SAGE_PALE, SAGE],
            labels={"revenue": "Revenue (€)", "county": "", "invoice_count": "Invoices"},
        )
        _styled_plotly_layout(fig)
        fig.update_xaxes(showgrid=True, gridcolor="#eee", tickformat=",.0f")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)


# Standalone test entry point
if __name__ == "__main__":
    st.set_page_config(page_title="Trade Manager - Dashboard", layout="wide")
    render_dashboard()
