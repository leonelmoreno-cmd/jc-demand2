#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# Demand Analysis JC — Google Trends + STL (LOESS)
# Fixed scope: US, last 5 years, en-US
# Best practices: retries/backoff, caching, validation, structured error messages.

import numpy as np
import pandas as pd
import streamlit as st

# --- Guard: detect urllib3>=2 (incompatible with pytrends 4.9.2 due to method_whitelist removal) ---
try:
    import urllib3
    from packaging import version
    if version.parse(urllib3.__version__) >= version.parse("2.0.0"):
        st.error(
            "Incompatible urllib3 version detected "
            f"({urllib3.__version__}). Please pin urllib3<2 in your environment:\n\n"
            "    pip install 'urllib3<2'\n\n"
            "This resolves the 'unexpected keyword argument method_whitelist' error."
        )
        st.stop()
except Exception:
    # If anything goes wrong with the check, continue (pytrends may still work if urllib3<2)
    pass

from pytrends.request import TrendReq
from statsmodels.tsa.seasonal import STL
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ---------- Streamlit basic setup ----------
st.set_page_config(page_title="Demand Analysis JC", layout="wide")
st.title("Demand Analysis JC")
st.caption("Google Trends (US, last 5y, en-US) → STL (LOESS) → Plotly")

# ---------- UI inputs ----------
kw = st.text_input("Keyword (required)", value="", placeholder="e.g., rocket stove")
run = st.button("Run analysis")

# Fixed config
HL = "en-US"            # interface language for Trends
TZ = 360                # timezone offset param (per pytrends examples)
TIMEFRAME = "today 5-y" # last 5 years
GEO = "US"              # United States

# ---------- Helpers ----------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_trends(keyword: str) -> pd.DataFrame:
    """Call Google Trends via pytrends and return a cleaned dataframe."""
    pytrends = TrendReq(
        hl=HL,
        tz=TZ,
        timeout=(10, 25),     # connect/read
        retries=2,
        backoff_factor=0.1,   # exponential backoff
    )
    pytrends.build_payload([keyword], timeframe=TIMEFRAME, geo=GEO)
    df = pytrends.interest_over_time()
    if df.empty:
        return df
    # Clean: drop last row and 'isPartial' per your workflow
    if len(df) > 0:
        df = df.iloc[:-1]
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def infer_period(dt_index: pd.DatetimeIndex) -> int:
    """Infer STL seasonal period from median sampling interval."""
    if len(dt_index) < 3:
        return 12
    deltas = np.diff(dt_index.values).astype("timedelta64[D]").astype(int)
    med = int(np.median(deltas))
    if med <= 1:
        return 7      # daily cadence -> weekly seasonality
    elif med <= 7:
        return 52     # weekly cadence -> yearly seasonality
    else:
        return 12     # monthly cadence -> yearly seasonality

def build_figure(df_plot: pd.DataFrame, title_kw: str) -> go.Figure:
    """Build 4-panel Plotly figure."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=("Original", "Trend", "Seasonal", "Residual")
    )
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["original"], name="Original", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["trend"],   name="Trend",   mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["seasonal"],name="Seasonal",mode="lines"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["remainder"],name="Residual",mode="lines"), row=4, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
    fig.update_layout(height=900, title_text=f"STL Decomposition — {title_kw} — Google Trends (US, last 5y)")
    return fig

# ---------- Execution ----------
if run:
    if not kw.strip():
        st.error("Please enter a keyword.")
        st.stop()

    with st.spinner("Fetching Google Trends…"):
        try:
            df = fetch_trends(kw.strip())
        except Exception as e:
            st.error(f"Error fetching data from Google Trends: {e}")
            st.info("Tip: pin urllib3<2 and try again if you see 'method_whitelist' errors.")
            st.stop()

    if df.empty:
        st.warning("No data returned by Google Trends for this keyword/timeframe/geo.")
        st.stop()

    # Series selection follows the exact column typed by user
    col_name = kw.strip()
    if col_name not in df.columns:
        st.error(f"Column '{col_name}' not found in Trends result.")
        st.stop()

    y = df[col_name].astype(float)
    period = infer_period(y.index)

    # STL (LOESS)
    try:
        res = STL(y, period=period, robust=True).fit()
    except Exception as e:
        st.error(f"STL decomposition failed: {e}")
        st.stop()

    # Build dataframe for plotting and CSV export
    df_plot = pd.DataFrame({
        "date": y.index,
        "original": y.values,
        "trend": res.trend,
        "seasonal": res.seasonal,
        "remainder": res.resid
    })

    # ---- Main decomposition figure ----
    fig = build_figure(df_plot, kw.strip())
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "*How to read this:* the **Trend** shows the long-term direction, **Seasonal** the recurring intra-year pattern, "
        "and **Residual** the short-term irregular component (noise/outliers). Values are relative Google Trends interest."
    )

    with st.expander("Show data (original/trend/seasonal/residual)"):
        st.dataframe(df_plot, use_container_width=True)
        st.download_button(
            "Download CSV",
            df_plot.to_csv(index=False).encode("utf-8"),
            file_name=f"stl_{kw.strip().replace(' ','_')}.csv",
            mime="text/csv"
        )

    # =========================
    # Additional analyses
    # =========================
    st.markdown("## Additional seasonal analyses")

    # ---- Chart 2: Average seasonal pattern by ISO week ----
    # Use ISO weeks (1–53) for consistency across years
    semana_mean = (
        df_plot.assign(iso_week=df_plot["date"].dt.isocalendar().week.astype(int))
               .groupby("iso_week", as_index=False)["seasonal"].mean()
               .rename(columns={"seasonal": "mean_seasonal"})
               .sort_values("iso_week")
               .reset_index(drop=True)
    )

    fig_week = px.line(
        semana_mean, x="iso_week", y="mean_seasonal",
        title="Average seasonal pattern by ISO week (1–53)"
    )
    fig_week.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_week.update_xaxes(dtick=4, title="ISO week (1–53)")
    fig_week.update_yaxes(title="Seasonal value")
    st.plotly_chart(fig_week, use_container_width=True)
    st.markdown(
        "*Interpretation:* this line shows the **average seasonal effect** at each ISO week across all years. "
        "Peaks indicate weeks that are typically above the baseline; troughs indicate below-baseline weeks."
    )

    # ---- Chart 3: Average seasonal pattern by month ----
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_map = dict(zip(range(1, 13), month_labels))

    mes_mean = (
        df_plot.assign(month_num=df_plot["date"].dt.month)
               .groupby("month_num", as_index=False)["seasonal"].mean()
               .rename(columns={"seasonal": "mean_seasonal"})
    )
    mes_mean["month_lab"] = mes_mean["month_num"].map(month_map)

    fig_month_mean = px.line(
        mes_mean, x="month_lab", y="mean_seasonal", markers=True,
        title="Average seasonal pattern by month"
    )
    fig_month_mean.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_month_mean.update_xaxes(title="Month")
    fig_month_mean.update_yaxes(title="Seasonal value")
    st.plotly_chart(fig_month_mean, use_container_width=True)
    st.markdown(
        "*Interpretation:* this summarizes the **typical monthly seasonality**. "
        "Use it to spot which months are usually stronger or weaker relative to the yearly baseline."
    )

    # ---- Chart 4: Seasonal distribution by month (box plot) ----
    df_box = df_plot.assign(
        month_num=df_plot["date"].dt.month,
        month_lab=lambda d: d["month_num"].map(month_map)
    )

    fig_box = px.box(
        df_box, x="month_lab", y="seasonal",
        title="Seasonal distribution by month"
    )
    fig_box.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_box.update_xaxes(title="Month")
    fig_box.update_yaxes(title="Seasonal value")
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown(
        "*Interpretation:* boxes show the **spread of seasonal values** for each month across years. "
        "Wider boxes or longer whiskers mean more variability; outliers capture unusual months."
    )

    # ---- Chart 5: Year-over-year seasonality comparison (last 3–4 years) ----
    df_plot = df_plot.copy()
    df_plot["year"] = df_plot["date"].dt.year
    df_plot["iso_week"] = df_plot["date"].dt.isocalendar().week.astype(int)

    # Determine a dynamic window covering the most recent 3–4 calendar years present in the data
    max_year = int(df_plot["year"].max())
    start_year = max(max_year - 3, int(df_plot["year"].min()))
    df_last_years = df_plot.query("@start_year <= year <= @max_year").copy()

    fig_yoy = px.line(
        df_last_years, x="iso_week", y="seasonal", color="year",
        title=f"Seasonality compared by year (from {start_year} to {max_year})",
        markers=False
    )
    fig_yoy.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_yoy.update_xaxes(title="ISO week", dtick=4)
    fig_yoy.update_yaxes(title="Seasonal value")
    st.plotly_chart(fig_yoy, use_container_width=True)
    st.markdown(
        "*Interpretation:* this compares **seasonal curves across recent years** on the same ISO-week axis. "
        "Look for alignment (stable seasonality) or divergences (shifts in timing or magnitude)."
    )

# ---------- Footer ----------
st.markdown(
    """
    <small>
    Data source: Google Trends via <code>pytrends</code> • Decomposition: <code>statsmodels.STL</code> • Charts: Plotly • Host: Streamlit Community Cloud
    </small>
    """, unsafe_allow_html=True
)

