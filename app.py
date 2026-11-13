"""
app.py
======
SustainScore AI — Production-Grade ESG Analytics Dashboard
Built with Streamlit + Plotly | Single source of truth: esg_engine.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import sys
import json
from datetime import datetime, date

import plotly.express as px
import plotly.graph_objects as go
import metadata_parser

from plot_setup import finastra_theme
from download_data import Data
import plots
import esg_engine
import recommendation_engine

# Advanced Intelligence Modules
from modules import prediction_engine
from modules import risk_engine
from modules import explainability_engine
from modules import narrative_engine
from modules import report_generator
from modules import graph_explainability
from modules import evaluation_engine
import streamlit.components.v1 as components

# ═══════════════════════════════════════════════════════════════════
#  CACHED FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data(start_data, end_data):
    data = Data().read(start_data, end_data)
    companies = data["data"].Organization.sort_values().unique().tolist()
    companies.insert(0, "Select a Company")
    return data, companies


@st.cache_data(show_spinner=False)
def load_company_profiles():
    try:
        with open("company_profiles.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@st.cache_data(show_spinner=False)
def filter_company_data(df_company, esg_categories, start, end):
    """Vectorized filtering."""
    if not esg_categories:
        return pd.DataFrame(columns=df_company.columns)

    existing_cats = [c for c in esg_categories if c in df_company.columns]
    if existing_cats:
        mask = df_company[existing_cats].any(axis=1)
        df_company = df_company[mask]

    df_company = df_company.copy()
    df_company["DATE"] = pd.to_datetime(df_company.DATE, dayfirst=True, format="mixed")
    df_company = df_company[df_company.DATE.dt.date.between(start, end)]
    return df_company


@st.cache_data(show_spinner=False)
def filter_publisher(df_company, publisher):
    if publisher != "all":
        df_company = df_company[df_company["SourceCommonName"] == publisher]
    return df_company


def filter_on_date(df, start, end, date_col="DATE"):
    if isinstance(start, date) and not isinstance(start, pd.Timestamp):
        start = pd.Timestamp(start)
    if isinstance(end, date) and not isinstance(end, pd.Timestamp):
        end = pd.Timestamp(end)
    if date_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, format="mixed")
    return df[(df[date_col] >= start) & (df[date_col] <= end)]


def get_melted_frame(data_dict, frame_names, keepcol=None, dropcol=None):
    reduced = {}
    for k, df in data_dict.items():
        if k not in frame_names:
            continue
        if keepcol:
            target = None
            if keepcol in df.columns:
                target = keepcol
            else:
                search_term = keepcol.lower().replace("_diff", "").replace("_", " ")
                for col in df.columns:
                    if col.lower().replace("_diff", "").replace("_", " ") == search_term:
                        target = col
                        break
            if target:
                reduced[k] = df[target].rename(k)
            else:
                reduced[k] = pd.Series(0, index=df.index).rename(k)
        else:
            cols_to_drop = [dropcol] if dropcol and dropcol in df.columns else []
            reduced[k] = df.drop(columns=cols_to_drop).mean(axis=1).rename(k)

    if not reduced:
        return pd.DataFrame(columns=["DATE", "ESG", "Score"])

    df = pd.concat(list(reduced.values()), axis=1).reset_index()
    actual_date_col = next((c for c in ["DATE", "date", "index"] if c in df.columns), df.columns[0])
    df = df.melt(actual_date_col).sort_values(actual_date_col).ffill()
    df.columns = ["DATE", "ESG", "Score"]
    return df.reset_index(drop=True)


def get_clickable_name(url):
    try:
        T = metadata_parser.MetadataParser(url=url, search_head_only=True)
        title = T.metadata["og"]["title"].replace("|", " - ")
        return f"[{title}]({url})"
    except Exception:
        return f"[{url}]({url})"


# ═══════════════════════════════════════════════════════════════════
#  EXPLANATION RENDERER
# ═══════════════════════════════════════════════════════════════════

def render_with_explanation(fig, meta):
    """Renders a Plotly or Altair figure with structured A-E explanation block."""
    if fig is not None and not isinstance(fig, bool):
        if hasattr(fig, "mark_line") or "altair" in str(type(fig)).lower():
            st.altair_chart(fig, use_container_width=True)
        else:
            try:
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("📊 Visualization currently unavailable.")
    elif fig is None:
        st.info("📊 No data available for this visualization.")

    st.markdown(f"""
        <div class="insight-box" style="font-size: 0.88em;">
            <b>A. Formula Used:</b> <br><i>{meta.get('formula', 'N/A')}</i><br><br>
            <b>B. What This Graph Represents:</b> <br>{meta.get('representation', 'N/A')}<br><br>
            <b>C. X-Axis:</b> <br>{meta.get('x_axis', 'N/A')}<br><br>
            <b>D. Y-Axis:</b> <br>{meta.get('y_axis', 'N/A')}<br><br>
            <b>E. Insight:</b> <br>{meta.get('insight', 'N/A')}
        </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  RECOMMENDATION RENDERER
# ═══════════════════════════════════════════════════════════════════

def render_recommendations(company, recommendations):
    """Renders the AI-driven recommendation system."""
    st.markdown(f"### 💡 ESG Improvement Recommendations for {company.title()}")

    # Summary
    summary = recommendations.get("summary", {})
    st.markdown(f"""
    <div class="insight-box" style="border-left-color: #F97316 !important;">
        {summary.get('priority_message', '')}<br><br>
        {summary.get('strength_message', '')}
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Pillar-by-pillar
    pillar_order = [
        ("overall", "📊 Overall ESG", "#6366F1"),
        ("environmental", "🌿 Environmental", "#10B981"),
        ("social", "👥 Social", "#3B82F6"),
        ("governance", "🏛️ Governance", "#F97316"),
    ]

    for key, title, color in pillar_order:
        rec_data = recommendations.get(key, {})
        severity = rec_data.get("severity", "strong")
        icon = rec_data.get("icon", "")
        severity_label = rec_data.get("severity_label", "")

        with st.expander(f"{title} — {icon} {severity_label}", expanded=(severity in ["critical", "watch"])):
            cols = st.columns([1, 1, 1])
            cols[0].metric("Company Score", f"{rec_data.get('company_display', 0):.1f}")
            cols[1].metric("Industry Ave", f"{rec_data.get('industry_display', 0):.1f}")
            cols[2].metric("Gap", f"{rec_data.get('gap_pts', 0):+.1f} pts")

            st.markdown("**Actionable Recommendations:**")
            for rec in rec_data.get("recommendations", []):
                st.markdown(f"  {rec}")


# ═══════════════════════════════════════════════════════════════════
#  RISK SENTINEL
# ═══════════════════════════════════════════════════════════════════

def render_risk_sentinel(company, df_company):
    """Detects and alerts on high-risk negative ESG events."""
    tone_col = "Tone" if "Tone" in df_company.columns else "AvgTone"
    if tone_col not in df_company.columns:
        return

    risk_df = df_company[df_company[tone_col] < -5.0].sort_values(tone_col).head(5)

    if not risk_df.empty:
        st.warning(f"🚨 Critical Risks detected for {company.title()}! {len(risk_df)} severe negative events found.")
        for _, row in risk_df.iterrows():
            with st.container():
                cols = st.columns([1, 4])
                cols[0].metric("Risk Score", f"{row[tone_col]:.1f}", delta_color="inverse")
                source = row.get("SourceCommonName", "Unknown")
                url = row.get("URL", "#")
                loc = row.get("Location", "N/A")
                dt = row.get("DATE", "N/A")
                cols[1].markdown(f"**[{source}]({url})**")
                cols[1].caption(f"📅 {dt} | 📍 {loc}")
    else:
        st.success(f"✅ No high-risk events detected for {company.title()} in the current filter.")


# ═══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════

def main(start_data, end_data):
    # Theme
    alt.themes.register("finastra", finastra_theme)
    alt.themes.enable("finastra")

    # Page Config
    icon_path = os.path.join(".", "raw", "esg_ai_logo.png")
    st.set_page_config(
        page_title="SustainScore AI | ESG Analytics",
        page_icon=icon_path,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ─── Premium CSS ───
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        :root {
            --bg-base: #FDFDFF;
            --accent-main: #4F46E5;
            --accent-glow: rgba(79, 70, 229, 0.15);
            --glass-bg: rgba(255, 255, 255, 0.65);
            --glass-border: rgba(0, 0, 0, 0.04);
            --text-title: #0F172A;
            --text-body: #334155;
            --text-muted: #64748B;
            --green: #10B981;
            --blue: #3B82F6;
            --orange: #F97316;
            --purple: #6366F1;
            --red: #EF4444;
        }

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            color: var(--text-body);
        }

        .stApp {
            background-color: var(--bg-base);
            background-image:
                radial-gradient(at 0% 0%, rgba(79, 70, 229, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(244, 114, 182, 0.08) 0px, transparent 50%),
                radial-gradient(at 0% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%);
            background-attachment: fixed;
        }

        [data-testid="stMetric"], .chart-container, .stExpander, .insight-box {
            background: var(--glass-bg) !important;
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border: 0.5px solid var(--glass-border) !important;
            border-radius: 20px !important;
            box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.02),
                        0 0 1px 0 rgba(0, 0, 0, 0.05) inset !important;
            transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1);
            padding: 1.5rem !important;
        }

        [data-testid="stMetric"] {
            min-height: 130px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
        }

        [data-testid="stMetric"]:hover, .chart-container:hover {
            transform: scale(1.01) translateY(-3px);
            border-color: rgba(79, 70, 229, 0.2) !important;
            box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.08) !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            color: var(--text-title) !important;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            font-size: 0.68rem !important;
            font-weight: 600 !important;
        }

        .hero-title {
            font-size: 4rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
            background: linear-gradient(135deg, #0F172A 30%, #4F46E5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.04em;
        }

        .hero-subtitle {
            color: var(--text-muted);
            font-size: 1.2rem;
            font-weight: 300;
            letter-spacing: 0.02em;
        }

        @keyframes reveal {
            from { opacity: 0; transform: translateY(20px); filter: blur(8px); }
            to { opacity: 1; transform: translateY(0); filter: blur(0); }
        }

        .main-content { animation: reveal 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards; }

        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.4) !important;
            backdrop-filter: blur(24px);
            border-right: 1px solid rgba(0, 0, 0, 0.05);
        }

        .insight-box {
            border-left: 4px solid var(--accent-main) !important;
            padding: 1.2rem !important;
            font-size: 0.88rem !important;
            line-height: 1.6 !important;
            margin-top: 0.8rem !important;
            background: rgba(79, 70, 229, 0.02) !important;
        }

        .severity-critical { border-left-color: var(--red) !important; background: rgba(239,68,68,0.04) !important; }
        .severity-watch { border-left-color: var(--orange) !important; background: rgba(249,115,22,0.04) !important; }
        .severity-strong { border-left-color: var(--green) !important; background: rgba(16,185,129,0.04) !important; }

        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            font-size: 0.9rem;
        }

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: rgba(0, 0, 0, 0.05);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover { background: rgba(79, 70, 229, 0.2); }
        </style>
    """, unsafe_allow_html=True)

    # ─── Hero Section ───
    st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem 1.5rem 1rem;">
            <h1 class="hero-title">SustainScore AI</h1>
            <p class="hero-subtitle">Production-Grade ESG Intelligence & Analytics Platform</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    #  SIDEBAR
    # ═══════════════════════════════════════════════════════════════

    st.sidebar.title("🧭 Navigation")
    st.sidebar.caption("v3.0 — Competition Edition")

    # Live GDELT
    if st.sidebar.button("🔄 Live GDELT"):
        with st.spinner("Fetching 15m GKG..."):
            try:
                import subprocess
                subprocess.run([sys.executable, "gdelt_fetcher.py", "--out", "Data/GDELT_Live"], check=True)
                st.sidebar.success("Live data updated!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Fetch failed: {e}")

    st.sidebar.markdown("---")

    # Dataset selection
    available_datasets = [d for d in os.listdir("Data") if os.path.isdir(os.path.join("Data", d))]
    default_idx = 0
    if "GDELT_20250101_20260329" in available_datasets:
        default_idx = available_datasets.index("GDELT_20250101_20260329")

    selected_ds = st.sidebar.selectbox("📁 Select Dataset", available_datasets, index=default_idx)

    if "_to_" in selected_ds:
        start_data, end_data = selected_ds.split("_to_")
    else:
        start_data, end_data = selected_ds, selected_ds

    # ESG Categories
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Filters")
    esg_categories = st.sidebar.multiselect("ESG Categories", ["E", "S", "G"], ["E", "S", "G"])
    pub = st.sidebar.empty()
    date_place = st.sidebar.empty()

    # ═══════════════════════════════════════════════════════════════
    #  LOAD DATA
    # ═══════════════════════════════════════════════════════════════

    try:
        with st.spinner(f"Loading dataset: {selected_ds}..."):
            data, companies = load_data(start_data, end_data)
    except Exception as e:
        st.error(f"⚠️ Error loading dataset '{selected_ds}': {e}")
        st.stop()

    df_data = data["data"]

    # ═══════════════════════════════════════════════════════════════
    #  COMPANY SELECTION
    # ═══════════════════════════════════════════════════════════════

    company = st.selectbox("🔍 Select a Company to Analyze", companies)

    if not company or company == "Select a Company":
        st.info("👆 Please select a company from the dropdown above to begin analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ═══════════════════════════════════════════════════════════════
    #  PREPARE COMPANY DATA
    # ═══════════════════════════════════════════════════════════════

    df_company = df_data[df_data.Organization.str.lower() == company.lower()].copy()

    if df_company.empty:
        st.warning(f"No data found for {company}.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Preprocess using the engine (single source of truth)
    df_company = esg_engine.preprocess_company_data(df_company)

    if df_company.empty:
        st.warning(f"No valid data after preprocessing for {company}.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ESG melted frames from pivot tables
    diff_col = f"{company.replace(' ', '_')}_diff"
    esg_keys = ["E_score", "S_score", "G_score"]
    esg_df = get_melted_frame(data, esg_keys, keepcol=diff_col)
    ind_esg_df = get_melted_frame(data, esg_keys, dropcol="industry_tone")
    tone_df = get_melted_frame(data, ["overall_score"], keepcol=diff_col)

    # Date widget
    start = df_company.DATE.min()
    end = df_company.DATE.max()
    if pd.isna(start):
        start = date.today()
    if pd.isna(end):
        end = date.today()

    selected_dates = date_place.date_input(
        "📅 Date Range", value=[start, end], min_value=start, max_value=end
    )

    if isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 2:
        start, end = selected_dates
    elif isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 1:
        start = end = selected_dates[0]
    else:
        start = end = selected_dates

    # Apply filters
    df_company = filter_company_data(df_company, esg_categories, start, end)
    esg_df = filter_on_date(esg_df, start, end)
    ind_esg_df = filter_on_date(ind_esg_df, start, end)
    tone_df = filter_on_date(tone_df, start, end)
    date_filtered_all = filter_on_date(df_data, start, end)

    # Publisher filter
    publishers = df_company.SourceCommonName.sort_values().unique().tolist() if not df_company.empty else []
    publishers.insert(0, "all")
    publisher = pub.selectbox("📰 Publisher Filter", publishers)
    df_company = filter_publisher(df_company, publisher)

    # Compute company summary with GLOBAL normalization
    # (normalizing across all companies ensures scores differentiate meaningfully)
    df_global_scored = esg_engine.compute_global_company_scores(date_filtered_all, company)
    company_summary = esg_engine.compute_company_summary(df_company, df_global_scored)
    exec_metrics = esg_engine.compute_executive_metrics(df_company, date_filtered_all)

    # ═══════════════════════════════════════════════════════════════
    #  TABBED DASHBOARD
    # ═══════════════════════════════════════════════════════════════

    tabs = st.tabs([
        "📊 Overview",
        "🎯 Individual Scores",
        "📈 Trends & Momentum",
        "📅 Monthly Analysis",
        "📅 Weekly Analysis",
        "🔮 AI Prediction",
        "⚠️ Risk Intelligence",
        "🧩 Score Breakdown",
        "📖 Performance Story",
        "🏆 Rankings & Comparison",
        "📉 Distributions & Outliers",
        "🔗 Correlations & Radar",
        "💡 Recommendations",
        "🔍 Advanced GDELT",
        "🕸️ Network XAI",
        "🔬 Academic Benchmarks",
        "📄 Download Report",
    ])

    # ═════════ TAB 1: OVERVIEW ═════════
    with tabs[0]:
        st.write("## Executive ESG Overview")

        # KPI Strip
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric(
            "Composite ESG",
            f"{exec_metrics['composite_esg']:.1f}",
            f"{exec_metrics['gap']:+.1f} pts vs Industry",
        )
        kc2.metric("Industry ESG", f"{exec_metrics['industry_esg']:.1f}")
        kc3.metric("Momentum", f"{exec_metrics['momentum']:+.3f}")
        kc4.metric("Volatility", f"{exec_metrics['volatility']:.3f}")

        st.write("###")

        # Company Profile
        profiles = load_company_profiles()
        profile_text = profiles.get(company.lower())
        if profile_text:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown(f"#### 🤖 {company.title()} — AI Analysis")
            st.write(profile_text)
            st.markdown('</div>', unsafe_allow_html=True)
            st.write("")

        # Risk Sentinel
        render_risk_sentinel(company, df_company)

        # Intelligence Summary
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown(f"### 🛡️ {company.title()} Intelligence Summary")

        tone_col = "Tone" if "Tone" in df_company.columns else "AvgTone"
        if tone_col in df_company.columns and not df_company.empty:
            sentiment_type = "positive" if df_company[tone_col].mean() > 0 else "cautious"
        else:
            sentiment_type = "neutral"

        # Find top ESG focus
        pillar_means = {
            "Environmental": company_summary["e_mean"],
            "Social": company_summary["s_mean"],
            "Governance": company_summary["g_mean"],
        }
        top_focus = max(pillar_means, key=pillar_means.get)

        sc1, sc2 = st.columns([2, 1])
        with sc1:
            st.write(f"""
            Based on analysis of **{company_summary['n_articles']}** articles,
            **{company.title()}** shows a **{sentiment_type}** sentiment trend
            (Trend: **{exec_metrics['trend']}**).
            The strongest pillar is **{top_focus}**, indicating this as the company's key ESG strength.
            """)
        with sc2:
            color = "#69db7c" if sentiment_type == "positive" else "#ff922b"
            st.markdown(f"""
                <div style="background-color: {color}20; padding: 15px; border-radius: 16px; border-left: 5px solid {color};">
                    <span style="color: {color}; font-weight: 700;">Status:</span> {sentiment_type.capitalize()}<br>
                    <span style="color: {color}; font-weight: 700;">Trend:</span> {exec_metrics['trend'].capitalize()}<br>
                    <span style="color: {color}; font-weight: 700;">Top Pillar:</span> {top_focus}
                </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # News Context
        st.write("---")
        with st.expander(f"📰 View {company.title()} News Context ({len(df_company):,d} articles)", expanded=False):
            display_cols = ["DATE", "Organization", "Person", "Location", "Tone", "SourceCommonName", "URL"]
            safe_cols = [c for c in display_cols if c in df_company.columns]

            csv = df_company.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Data (CSV)", csv, f"{company.lower()}_esg_data.csv", "text/csv")
            st.dataframe(df_company[safe_cols], use_container_width=True)

    # ═════════ TAB 2: INDIVIDUAL SCORES ═════════
    with tabs[1]:
        st.write(f"### 🎯 {company.title()} — Pillar Breakdown & Detailed Methodology")

        # Score cards
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Overall ESG", f"{company_summary['overall_mean']:.3f}")
        sc2.metric("Environmental (E)", f"{company_summary['e_mean']:.3f}")
        sc3.metric("Social (S)", f"{company_summary['s_mean']:.3f}")
        sc4.metric("Governance (G)", f"{company_summary['g_mean']:.3f}")

        st.write("---")

        # Individual pillar charts (3 columns)
        p_col1, p_col2, p_col3 = st.columns(3)

        with p_col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_e = plots.plot_individual_pillar(df_company, company, "Environmental")
            render_with_explanation(fig_e, {
                'formula': esg_engine.FORMULAS['E_score'],
                'representation': 'Environmental pillar tracking media volume and tone. Higher volume with positive tone indicates strong environmental disclosure.',
                'x_axis': 'Date — timeline of media events.',
                'y_axis': 'E_score — Z-score normalized Environmental score.',
                'insight': f'Shows <b>{company.title()}</b>\'s environmental footprint visibility. Spikes indicate major environmental events or reports.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

        with p_col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_s = plots.plot_individual_pillar(df_company, company, "Social")
            render_with_explanation(fig_s, {
                'formula': esg_engine.FORMULAS['S_score'],
                'representation': 'Social engagement balanced between mention breadth and sentiment quality.',
                'x_axis': 'Date — timeline of media events.',
                'y_axis': 'S_score — Z-score normalized Social score.',
                'insight': f'Highlights <b>{company.title()}</b>\'s societal impact. High scores suggest positive community and employee relations.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

        with p_col3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_g = plots.plot_individual_pillar(df_company, company, "Governance")
            render_with_explanation(fig_g, {
                'formula': esg_engine.FORMULAS['G_score'],
                'representation': 'Governance score heavily weighted towards sentiment stability and ethics.',
                'x_axis': 'Date — timeline of media events.',
                'y_axis': 'G_score — Z-score normalized Governance score.',
                'insight': f'Indicates <b>{company.title()}</b>\'s leadership stability, ethics, and board integrity in the media.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

        # Overall ESG chart
        st.write("#### Overall Composite ESG Score")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_overall = plots.plot_individual_pillar(df_company, company, "Overall")
        render_with_explanation(fig_overall, {
            'formula': esg_engine.FORMULAS['Overall_ESG'],
            'representation': 'Weighted composite of E, S, G pillars providing a holistic sustainability view.',
            'x_axis': 'Date — timeline of media events.',
            'y_axis': 'Overall ESG Score — weighted Z-score normalized composite.',
            'insight': f'Combines all three pillars for <b>{company.title()}</b>. Stable positive values indicate consistent ESG leadership.',
        })
        st.markdown('</div>', unsafe_allow_html=True)

        # Weighted waterfall
        st.write("#### Weighted Contribution Breakdown")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_wf = plots.plot_weighted_esg_waterfall(company_summary, company)
        render_with_explanation(fig_wf, {
            'formula': f'E×{esg_engine.PILLAR_WEIGHTS["E_score"]} + S×{esg_engine.PILLAR_WEIGHTS["S_score"]} + G×{esg_engine.PILLAR_WEIGHTS["G_score"]}',
            'representation': 'Waterfall chart showing each pillar\'s weighted contribution to the total ESG score.',
            'x_axis': 'ESG Component (with weight factor).',
            'y_axis': 'Weighted score contribution.',
            'insight': f'Shows how each pillar contributes to <b>{company.title()}</b>\'s total ESG score. Negative contributions pull the score down.',
        })
        st.markdown('</div>', unsafe_allow_html=True)

        # ESG Pillar Comparison Bar
        st.write("#### Side-by-Side Pillar Comparison")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_comp = plots.plot_esg_pillar_comparison(df_company, company)
        render_with_explanation(fig_comp, {
            'formula': 'Mean of E_score, S_score, and G_score respectively.',
            'representation': 'Bar chart comparing absolute E, S, and G mean scores side-by-side.',
            'x_axis': 'ESG Pillar category.',
            'y_axis': 'Mean normalized score for the selected period.',
            'insight': f'Quick comparison showing which pillar is strongest for <b>{company.title()}</b>.',
        })
        st.markdown('</div>', unsafe_allow_html=True)

    # ═════════ TAB 3: TRENDS & MOMENTUM ═════════
    with tabs[2]:
        st.write(f"### 📈 {company.title()} — Trend Analysis & Momentum")

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_trend = plots.plot_esg_trend(df_company, company)
            render_with_explanation(fig_trend, {
                'formula': f'{esg_engine.FORMULAS["E_score"]}<br>{esg_engine.FORMULAS["S_score"]}<br>{esg_engine.FORMULAS["G_score"]}',
                'representation': 'Combined line chart tracking all three ESG pillars over time.',
                'x_axis': 'Date timeline.',
                'y_axis': 'Standardized score. Higher = stronger ESG signal.',
                'insight': f'Reveals the trajectory and volatility of each pillar for <b>{company.title()}</b>.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

        with col_t2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_ta = plots.plot_trend_analysis(df_company, company)
            render_with_explanation(fig_ta, {
                'formula': 'Moving Average: MA(k) = (1/k) × Σ ESG(t-i) for i=0..k-1',
                'representation': 'Overall ESG with a smoothed moving average to identify long-term trends.',
                'x_axis': 'Date timeline.',
                'y_axis': 'Overall ESG Score.',
                'insight': f'The moving average smooths noise. If the MA is rising, <b>{company.title()}</b>\'s ESG is improving over time.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("---")

        col_t3, col_t4 = st.columns(2)

        with col_t3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_mom = plots.plot_sentiment_momentum(tone_df, company)
            render_with_explanation(fig_mom, {
                'formula': 'Momentum = μ_recent(20%) − μ_baseline(20%)',
                'representation': 'Volume-weighted momentum showing changes in sentiment over time.',
                'x_axis': 'Date timeline.',
                'y_axis': 'Sentiment Score.',
                'insight': f'Positive momentum indicates improving public perception for <b>{company.title()}</b>.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

        with col_t4:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_vol = plots.plot_sentiment_volatility(df_company, company)
            render_with_explanation(fig_vol, {
                'formula': 'σ_rolling = Rolling StdDev of Normalized Tone',
                'representation': 'Line chart showing the magnitude of sentiment fluctuations.',
                'x_axis': 'Date.',
                'y_axis': 'Volatility Index (rolling std dev).',
                'insight': 'Spikes indicate media instability or controversy. Stable zones reflect consistent messaging.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

    # ═════════ TAB 4: MONTHLY ANALYSIS ═════════
    with tabs[3]:
        st.write(f"### 📅 {company.title()} — Monthly ESG Analysis")
        
        # 1. Standardized Aggregation (Using Industry-Benchmarked Global Scores)
        df_monthly = esg_engine.compute_esg_aggregates(df_global_scored, freq='monthly')
        df_monthly = esg_engine.compute_growth_metrics(df_monthly)
        df_monthly = esg_engine.detect_esg_anomalies(df_monthly)
        
        if df_monthly.empty:
            st.info("Not enough monthly data to display analysis. Ensure the date range covers at least one full month.")
        else:
            # Monthly Selector
            df_monthly['Month_Label'] = df_monthly['DATE'].dt.strftime('%b %Y')
            month_options = df_monthly['Month_Label'].tolist()
            selected_month_label = st.selectbox("Select Month to Inspect", month_options, index=len(month_options)-1)
            
            row_month = df_monthly[df_monthly['Month_Label'] == selected_month_label].iloc[0]
            
            # Monthly KPI Cards
            mc1, mc2, mc3, mc4 = st.columns(4)
            
            def render_growth_metric(col, label, val, growth_pct, delta, stat_var):
                color = "normal" if delta > 0 else "inverse"
                col.metric(label, f"{val:.3f}", f"{delta:+.3f} Δ ({growth_pct:+.1f}%)", delta_color=color)
                # User-friendly stability indicator
                if stat_var < 0.01:
                    stability = "🟢 Very Stable"
                elif stat_var < 0.05:
                    stability = "🟢 Stable"
                elif stat_var < 0.15:
                    stability = "🟡 Moderate Fluctuation"
                elif stat_var < 0.30:
                    stability = "🟠 High Fluctuation"
                else:
                    stability = "🔴 Highly Volatile"
                col.caption(f"Consistency: {stability}")

            render_growth_metric(mc1, "Overall ESG", row_month['Overall_ESG'], row_month['Overall_ESG_Growth_Pct'], row_month['Overall_ESG_Performance_Delta'], row_month['Overall_ESG_Stat_Var'])
            render_growth_metric(mc2, "Environmental", row_month['E_score'], row_month['E_score_Growth_Pct'], row_month['E_score_Performance_Delta'], row_month['E_score_Stat_Var'])
            render_growth_metric(mc3, "Social", row_month['S_score'], row_month['S_score_Growth_Pct'], row_month['S_score_Performance_Delta'], row_month['S_score_Stat_Var'])
            render_growth_metric(mc4, "Governance", row_month['G_score'], row_month['G_score_Growth_Pct'], row_month['G_score_Performance_Delta'], row_month['G_score_Stat_Var'])
            
            st.write("---")
            
            st.write(f"#### 🔎 Detailed Monthly Breakdown: {selected_month_label}")
            st.info("The following charts show daily ESG variations *within* the selected month to identify specific high-impact days and potential risks.")

            # Filter original daily data for the selected month (Using Global Scores)
            target_date = pd.to_datetime(selected_month_label, format='%b %Y')
            df_selected_month = df_global_scored[
                (df_global_scored['DATE'].dt.year == target_date.year) & 
                (df_global_scored['DATE'].dt.month == target_date.month)
            ].sort_values("DATE")

            if df_selected_month.empty:
                st.warning("Daily granularity not available for this month. Showing monthly aggregate only.")
            else:
                # 4 Focused Charts: E, S, G, Overall
                fcol1, fcol2 = st.columns(2)
                
                with fcol1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_fe = plots.plot_focused_variation(df_selected_month, company, selected_month_label, "Environmental")
                    render_with_explanation(fig_fe, {
                        'formula': 'Daily Environmental (E) sub-score variation.',
                        'representation': 'Focused line chart of E-pillar performance across the month.',
                        'x_axis': f'Days in {selected_month_label}.',
                        'y_axis': 'E-Score (Normalized).',
                        'insight': 'Identifies specific days where environmental sentiment peaked or dropped. The dashed line shows the monthly average.',
                    })
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_fs = plots.plot_focused_variation(df_selected_month, company, selected_month_label, "Social")
                    render_with_explanation(fig_fs, {
                        'formula': 'Daily Social (S) sub-score variation.',
                        'representation': 'Focused line chart of S-pillar performance across the month.',
                        'x_axis': f'Days in {selected_month_label}.',
                        'y_axis': 'S-Score (Normalized).',
                        'insight': 'Captures daily shifts in social sentiment, workforce news, or community impact reports.',
                    })
                    st.markdown('</div>', unsafe_allow_html=True)

                with fcol2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_fg = plots.plot_focused_variation(df_selected_month, company, selected_month_label, "Governance")
                    render_with_explanation(fig_fg, {
                        'formula': 'Daily Governance (G) sub-score variation.',
                        'representation': 'Focused line chart of G-pillar performance across the month.',
                        'x_axis': f'Days in {selected_month_label}.',
                        'y_axis': 'G-Score (Normalized).',
                        'insight': 'Tracks daily governance-related news, leadership sentiment, or regulatory mentions.',
                    })
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_fo = plots.plot_focused_variation(df_selected_month, company, selected_month_label, "Overall")
                    render_with_explanation(fig_fo, {
                        'formula': 'Daily Composite ESG Score variation.',
                        'representation': 'Total ESG intelligence view focusing on the selected month.',
                        'x_axis': f'Days in {selected_month_label}.',
                        'y_axis': 'Overall ESG Score.',
                        'insight': 'The cumulative daily ESG pulse. Green triangles mark the monthly peak, red mark the lowest point.',
                    })
                    st.markdown('</div>', unsafe_allow_html=True)

            st.write("---")
            
            # Anomaly Alerts
            anomalies = df_monthly[df_monthly['is_anomaly']]
            if not anomalies.empty:
                st.warning(f"⚠️ {len(anomalies)} Monthly Historical Anomalies Detected!")
                for _, anon in anomalies.iterrows():
                    st.write(f"• **{anon['Month_Label']}**: Significant {anon['anomaly_type']} (Z={anon['z_score']:.2f})")
            
            # Forward-looking Strategy
            st.write(f"#### 🚀 ESG Strategy & Next Month Improvements: {selected_month_label}")
            
            # Identify weakest pillar
            pillars_row = {
                "Environmental": row_month['E_score'],
                "Social": row_month['S_score'],
                "Governance": row_month['G_score']
            }
            weakest_pillar = min(pillars_row, key=pillars_row.get)
            
            st.markdown(f"""
            <div class="insight-box">
            <b>Targeted Analysis:</b> For the month of {selected_month_label}, the weakest component is <b>{weakest_pillar}</b>. 
            To improve performance in the <b>NEXT MONTH</b>, consider the following actions:
            </div>
            """, unsafe_allow_html=True)

            # Monthly recommendations
            industry_bench = esg_engine.compute_industry_benchmark(date_filtered_all)
            month_scores = {
                "e_mean": row_month['E_score'],
                "s_mean": row_month['S_score'],
                "g_mean": row_month['G_score'],
                "overall_mean": row_month['Overall_ESG'],
            }
            month_recs = recommendation_engine.generate_recommendations(month_scores, industry_bench)
            
            render_recommendations(company, month_recs)
            
            # Comparison Bar Chart (Move to bottom)
            st.write("#### Historical Context: Monthly Performance Comparison")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_m_comp = plots.plot_period_comparison(df_monthly, company, "Month")
            st.plotly_chart(fig_m_comp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ═════════ TAB 5: WEEKLY ANALYSIS ═════════
    with tabs[4]:
        st.write(f"### 📅 {company.title()} — Weekly ESG Analysis")

        # 1. Standardized Aggregation (Using Industry-Benchmarked Global Scores)
        df_weekly = esg_engine.compute_esg_aggregates(df_global_scored, freq='weekly')
        df_weekly = esg_engine.compute_growth_metrics(df_weekly)
        df_weekly = esg_engine.detect_esg_anomalies(df_weekly)

        if df_weekly.empty:
            st.info("Not enough weekly data to display analysis. Ensure the date range spans at least two weeks.")
        else:
            # Weekly Selector
            df_weekly['iso_year'] = df_weekly['DATE'].dt.isocalendar().year
            df_weekly['iso_week'] = df_weekly['DATE'].dt.isocalendar().week
            df_weekly['Week_Label'] = df_weekly.apply(lambda row: f"Week {row['iso_week']:02d}, {row['iso_year']}", axis=1)
            week_options = df_weekly['Week_Label'].tolist()
            selected_week_label = st.selectbox("Select Week to Inspect", week_options, index=len(week_options)-1)

            row_week = df_weekly[df_weekly['Week_Label'] == selected_week_label].iloc[0]

            # Weekly KPI Cards
            wc1, wc2, wc3, wc4 = st.columns(4)

            def render_w_growth_metric(col, label, val, growth_pct, delta, stat_var):
                color = "normal" if delta > 0 else "inverse"
                col.metric(label, f"{val:.3f}", f"{delta:+.3f} Δ ({growth_pct:+.1f}%)", delta_color=color)
                # User-friendly stability indicator
                if stat_var < 0.01:
                    stability = "🟢 Very Stable"
                elif stat_var < 0.05:
                    stability = "🟢 Stable"
                elif stat_var < 0.15:
                    stability = "🟡 Moderate Fluctuation"
                elif stat_var < 0.30:
                    stability = "🟠 High Fluctuation"
                else:
                    stability = "🔴 Highly Volatile"
                col.caption(f"Score Consistency: {stability}")

            render_w_growth_metric(wc1, "Overall ESG", row_week['Overall_ESG'], row_week['Overall_ESG_Growth_Pct'], row_week['Overall_ESG_Performance_Delta'], row_week['Overall_ESG_Stat_Var'])
            render_w_growth_metric(wc2, "Environmental", row_week['E_score'], row_week['E_score_Growth_Pct'], row_week['E_score_Performance_Delta'], row_week['E_score_Stat_Var'])
            render_w_growth_metric(wc3, "Social", row_week['S_score'], row_week['S_score_Growth_Pct'], row_week['S_score_Performance_Delta'], row_week['S_score_Stat_Var'])
            render_w_growth_metric(wc4, "Governance", row_week['G_score'], row_week['G_score_Growth_Pct'], row_week['G_score_Performance_Delta'], row_week['G_score_Stat_Var'])

            st.write("---")

            st.write(f"#### 🔎 Detailed Weekly Breakdown: {selected_week_label}")
            st.info("The following charts show daily ESG variations during the selected week. High-frequency tracking allows for immediate response to specific events.")

            # Filter original daily data for the selected week (Using Global Scores)
            selected_iso_year = row_week['iso_year']
            selected_iso_week = row_week['iso_week']
            df_selected_week = df_global_scored[
                (df_global_scored['DATE'].dt.isocalendar().year == selected_iso_year) & 
                (df_global_scored['DATE'].dt.isocalendar().week == selected_iso_week)
            ].sort_values("DATE")

            if df_selected_week.empty:
                st.warning("Daily granularity not available for this week. Showing weekly aggregate only.")
            else:
                # 4 Focused Charts: E, S, G, Overall
                wcol1, wcol2 = st.columns(2)
                
                with wcol1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_we = plots.plot_focused_variation(df_selected_week, company, selected_week_label, "Environmental")
                    render_with_explanation(fig_we, {
                        'formula': 'Daily Environmental (E) sub-score variation.',
                        'representation': 'Focused line chart of E-pillar performance across the week.',
                        'x_axis': f'Days in {selected_week_label}.',
                        'y_axis': 'E-Score (Normalized).',
                        'insight': 'Captures high-frequency shifts in environmental sentiment. Useful for identifying immediate impact of specific news.',
                    })
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_ws = plots.plot_focused_variation(df_selected_week, company, selected_week_label, "Social")
                    render_with_explanation(fig_ws, {
                        'formula': 'Daily Social (S) sub-score variation.',
                        'representation': 'Focused line chart of S-pillar performance across the week.',
                        'x_axis': f'Days in {selected_week_label}.',
                        'y_axis': 'S-Score (Normalized).',
                        'insight': 'Tracks immediate social sentiment changes related to workforce or community news.',
                    })
                    st.markdown('</div>', unsafe_allow_html=True)

                with wcol2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_wg = plots.plot_focused_variation(df_selected_week, company, selected_week_label, "Governance")
                    render_with_explanation(fig_wg, {
                        'formula': 'Daily Governance (G) sub-score variation.',
                        'representation': 'Focused line chart of G-pillar performance across the week.',
                        'x_axis': f'Days in {selected_week_label}.',
                        'y_axis': 'G-Score (Normalized).',
                        'insight': 'Identifies specific days with governance-related mentions or leadership activity.',
                    })
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig_wo = plots.plot_focused_variation(df_selected_week, company, selected_week_label, "Overall")
                    render_with_explanation(fig_wo, {
                        'formula': 'Daily Composite ESG Score variation.',
                        'representation': 'Total ESG pulse view focusing on the selected week.',
                        'x_axis': f'Days in {selected_week_label}.',
                        'y_axis': 'Overall ESG Score.',
                        'insight': 'The daily composite score. Green/Red markers highlight the best and worst performing days of the week.',
                    })
                    st.markdown('</div>', unsafe_allow_html=True)

            st.write("---")

            # Growth Summary
            perf_summary = esg_engine.get_period_performance_summary(df_weekly)
            sc1, sc2 = st.columns(2)
            with sc1:
                st.success(f"🌟 **Best Performing Period**: {perf_summary['best_period'].strftime('%Y-%m-%d')} (Score: {perf_summary['best_score']:.3f})")
            with sc2:
                st.error(f"⚠️ **Worst Performing Period**: {perf_summary['worst_period'].strftime('%Y-%m-%d')} (Score: {perf_summary['worst_score']:.3f})")

            # Forward-looking Strategy
            st.write(f"#### 🚀 ESG Strategy & Next Week Improvements: {selected_week_label}")
            
            # Identify weakest pillar
            pillars_row_w = {
                "Environmental": row_week['E_score'],
                "Social": row_week['S_score'],
                "Governance": row_week['G_score']
            }
            weakest_pillar_w = min(pillars_row_w, key=pillars_row_w.get)
            
            st.markdown(f"""
            <div class="insight-box">
            <b>Targeted Analysis:</b> For the week of {selected_week_label}, the weakest component is <b>{weakest_pillar_w}</b>. 
            To improve performance in the <b>NEXT WEEK</b>, consider the following actions:
            </div>
            """, unsafe_allow_html=True)

            week_scores_w = {
                "e_mean": row_week['E_score'],
                "s_mean": row_week['S_score'],
                "g_mean": row_week['G_score'],
                "overall_mean": row_week['Overall_ESG'],
            }
            week_recs_w = recommendation_engine.generate_recommendations(week_scores_w, industry_bench)
            render_recommendations(company, week_recs_w)
            
            # Historical Comparison (Move to bottom)
            st.write("#### Historical Context: Weekly Performance Comparison")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_w_comp = plots.plot_period_comparison(df_weekly, company, "Week")
            st.plotly_chart(fig_w_comp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ═════════ TAB 6: 🔮 AI PREDICTION ═════════
    with tabs[5]:
        st.write(f"### 🔮 {company.title()} — AI ESG Score Prediction")
        
        target_col = st.selectbox("Select Target Benchmark", 
                                ["Overall_ESG", "E_score", "S_score", "G_score"], 
                                key="pred_target")
        
        # Prepare historical data (Using Industry-Benchmarked Global Scores)
        df_hist = esg_engine.compute_esg_aggregates(df_global_scored, freq='weekly')
        df_pred = prediction_engine.predict_future_scores(df_hist, periods=4, column=target_col)
        
        if not df_pred.empty:
            pct_change, sentiment = prediction_engine.get_forecast_summary(df_hist, df_pred, column=target_col)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Forecasted Score", f"{df_pred['Predicted_Score'].iloc[-1]:.3f}", f"{pct_change:+.1f}%")
            with c2:
                st.metric("Trend Sentiment", sentiment.title())
            with c3:
                st.metric("Confidence Level", "95% (H-CI)")

            # Visualization
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_pred = go.Figure()
            
            # Historical
            fig_pred.add_trace(go.Scatter(
                x=df_hist['DATE'], y=df_hist[target_col],
                mode='lines+markers', name='Historical',
                line=dict(color='#4F46E5', width=3)
            ))
            
            # Forecast
            fig_pred.add_trace(go.Scatter(
                x=df_pred['DATE'], y=df_pred['Predicted_Score'],
                mode='lines+markers', name='Forecast',
                line=dict(color='#10B981', width=4, dash='dash')
            ))
            
            # Confidence Interval
            fig_pred.add_trace(go.Scatter(
                x=pd.concat([df_pred['DATE'], df_pred['DATE'][::-1]]),
                y=pd.concat([df_pred['Upper_CI'], df_pred['Lower_CI'][::-1]]),
                fill='toself', fillcolor='rgba(16, 185, 129, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip", showlegend=True, name='95% Confidence'
            ))
            
            fig_pred.update_layout(
                template='plotly_white',
                title=f"AI Forecast: {target_col} Projection",
                xaxis_title="Date", yaxis_title="ESG Score",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            render_with_explanation(fig_pred, {
                'formula': 'Linear Regression Projection: y = β₀ + β₁x + ε',
                'representation': 'Combines historical daily momentum with a 4-week forward-looking trend window.',
                'x_axis': 'Timeline (Weekly aggregated bands).',
                'y_axis': 'Normalized ESG score (Historical vs Projected).',
                'insight': f'The model predicts a <b>{sentiment}</b> trend. If the score is above the confidence band, the company is outperforming its news cycle momentum.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

    # ═════════ TAB 7: ⚠️ RISK INTELLIGENCE ═════════
    with tabs[6]:
        st.write(f"### ⚠️ {company.title()} — ESG Risk Intelligence")
        
        risks = []
        for col in ["Overall_ESG", "E_score", "S_score", "G_score"]:
            risks.extend(risk_engine.detect_risks(df_company, column=col))
            
        if risks:
            for risk in risks:
                icon = "🚨" if risk['severity'] == "Critical" else "⚠️"
                with st.expander(f"{icon} {risk['type']} - {risk['pillar']} Pillar ({risk['severity']})", expanded=True):
                    st.write(risk['message'])
                    st.progress(1.0 if risk['severity'] == "Critical" else 0.5)
        else:
            st.success("No critical structural ESG risks detected in the current monitoring window.")
            st.info("Continuous monitoring active: Sentiment volatility and trend slopes are within normal historical bounds.")

        # Risk Timeline
        st.write("#### Risk Volatility Map")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_vol = px.line(df_company, x="DATE", y="Overall_ESG", title="Real-time Volatility Monitoring")
        # Highlight anomalies
        anoms = esg_engine.detect_esg_anomalies(esg_engine.compute_esg_aggregates(df_company, 'weekly'))
        if not anoms.empty:
            anom_pts = anoms[anoms['is_anomaly']]
            fig_vol.add_trace(go.Scatter(
                x=anom_pts['DATE'], y=anom_pts['Overall_ESG'],
                mode='markers', name='Anomalies',
                marker=dict(color='red', size=12, symbol='x')
            ))
        
        render_with_explanation(fig_vol, {
            'formula': 'Z-Score Anomaly: |(val - mean)| / std > 2.0',
            'representation': 'Volatility map identifying periods where ESG performance deviated significantly from the historical mean.',
            'x_axis': 'Date', 'y_axis': 'Overall ESG Score',
            'insight': 'Red X marks indicate "ESG Shocks" — days with news events that caused statistical disruption.',
        })
        st.markdown('</div>', unsafe_allow_html=True)

    # ═════════ TAB 8: 🧩 SCORE BREAKDOWN ═════════
    with tabs[7]:
        st.write(f"### 🧩 {company.title()} — Multi-Factor Score Breakdown")
        
        sel_pillar = st.radio("Focus Area", ["Overall_ESG", "E_score", "S_score", "G_score"], horizontal=True)
        factors = explainability_engine.get_factor_percentages(sel_pillar, scores=company_summary)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            labels = [f[0] for f in factors]
            values = [f[1] for f in factors]
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=['#4F46E5', '#10B981', '#F97316'])])
            fig_pie.update_layout(title="Contribution Weights")
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.write("#### Factor Intelligence")
            for f_name, f_pct in factors:
                st.write(f"**{f_name}** ({f_pct}%)")
                st.caption("Derived from article density and secondary keyword sentiment clustering.")
                st.progress(f_pct / 100)

        st.info("NOTE: Breakdown percentages are weighted based on GDELT intelligence inputs (NumArticles vs AvgTone) mapped to industry-standard sub-pillars.")

    # ═════════ TAB 9: 📖 PERFORMANCE STORY ═════════
    with tabs[8]:
        st.write(f"### 📖 {company.title()} — ESG Performance Narrative")
        
        df_res = esg_engine.compute_esg_aggregates(df_company, 'monthly')
        story_points = narrative_engine.generate_performance_story(df_res)
        
        if story_points:
            st.write(narrative_engine.format_story_as_text(story_points))
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_story = px.line(df_res, x="DATE", y="Overall_ESG", title="Annotated ESG Journey")
            for pt in story_points:
                fig_story.add_annotation(
                    x=pt['DATE'], y=pt['Score'],
                    text=pt['Event'], showarrow=True, arrowhead=2,
                    bgcolor="#4F46E5", font=dict(color="white")
                )
            
            render_with_explanation(fig_story, {
                'formula': 'Event detection via local maxima/minima and momentum shifts.',
                'representation': 'An automated visual narrative highlighting the defining moments of the company\'s ESG journey.',
                'x_axis': 'Time', 'y_axis': 'Performance',
                'insight': 'This story summarizes major turning points. Use this to explain performance to stakeholders.',
            })
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Timeline story requires a longer historical window to generate meaningful event highlights.")

    # ═════════ TAB 10: RANKINGS & COMPARISON ═════════
    with tabs[9]:
        st.write("### 🏆 Company Rankings & Cross-Company Comparison")

        # Company Ranking
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_rank = plots.plot_company_ranking(date_filtered_all, company)
        render_with_explanation(fig_rank, {
            'formula': esg_engine.FORMULAS['Overall_ESG'] + ' → scaled to 0-100',
            'representation': 'Horizontal bar chart ranking companies by composite ESG score.',
            'x_axis': 'ESG Score (0-100 scale).',
            'y_axis': 'Company name with rank.',
            'insight': f'Shows where <b>{company.title()}</b> stands relative to peers. Highlighted in purple.',
        })
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("---")

        # Best & Worst
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_bw = plots.plot_best_worst_companies(date_filtered_all, n=5)
        render_with_explanation(fig_bw, {
            'formula': 'Top/Bottom 5 companies by mean Overall_ESG score.',
            'representation': 'Dual bar chart highlighting the best and worst ESG performers.',
            'x_axis': 'ESG Score (0-100 scale).',
            'y_axis': 'Company name.',
            'insight': 'Green bars are top performers — use them as benchmarks. Red bars indicate companies needing improvement.',
        })
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("---")

        # Multi-company comparison
        st.write("#### 📊 Multi-Company Comparison")
        all_companies = sorted(df_data["Organization"].unique().tolist())
        default_compare = [company]

        selected_for_compare = st.multiselect(
            "Select companies to compare (up to 5)",
            all_companies,
            default=default_compare,
            max_selections=5,
        )

        if selected_for_compare:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_mc = plots.plot_multi_company_comparison(date_filtered_all, selected_for_compare)
            render_with_explanation(fig_mc, {
                'formula': 'Mean E, S, G scores per company.',
                'representation': 'Grouped bar chart comparing ESG pillars across selected companies.',
                'x_axis': 'ESG Pillar.',
                'y_axis': 'Mean normalized score.',
                'insight': 'Direct comparison reveals relative strengths and weaknesses of each company per pillar.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

        # ESG Gap Comparison (Company vs Industry)
        st.write("---")
        st.write("#### 📏 ESG Gap vs Industry")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        industry_bench = esg_engine.compute_industry_benchmark(date_filtered_all)
        categories = ["Environmental", "Social", "Governance"]
        company_vals = [company_summary["e_mean"], company_summary["s_mean"], company_summary["g_mean"]]
        industry_vals = [industry_bench["e_mean"], industry_bench["s_mean"], industry_bench["g_mean"]]
        gaps = [c - i for c, i in zip(company_vals, industry_vals)]

        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(
            x=categories, y=company_vals, name=company.title(),
            marker_color="#4F46E5", text=[f"{v:.3f}" for v in company_vals], textposition="outside",
        ))
        fig_gap.add_trace(go.Bar(
            x=categories, y=industry_vals, name="Industry",
            marker_color="#94A3B8", text=[f"{v:.3f}" for v in industry_vals], textposition="outside",
        ))
        gap_colors = ["#10B981" if g > 0 else "#EF4444" for g in gaps]
        fig_gap.add_trace(go.Bar(
            x=categories, y=gaps, name="Gap",
            marker_color=gap_colors, text=[f"{g:+.3f}" for g in gaps], textposition="outside",
        ))
        fig_gap.update_layout(
            barmode="group", title=f"{company.title()} vs Industry ESG Performance",
            template="simple_white", height=460, yaxis_title="Score",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        )

        render_with_explanation(fig_gap, {
            'formula': 'Gap = Company Score − Industry Average Score per pillar.',
            'representation': 'Grouped bar chart showing company vs industry with gap overlay.',
            'x_axis': 'ESG Pillar category.',
            'y_axis': 'Mean Score (company, industry, and gap).',
            'insight': f'Green gaps mean <b>{company.title()}</b> outperforms industry. Red gaps highlight risk areas.',
        })
        st.markdown('</div>', unsafe_allow_html=True)

    # ═════════ TAB 11: DISTRIBUTIONS & OUTLIERS ═════════
    with tabs[10]:
        st.write("### 📉 Score Distributions & Outlier Detection")

        # Box plots
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_box = plots.plot_esg_distribution_boxplot(date_filtered_all)
        render_with_explanation(fig_box, {
            'formula': 'Per-company mean of each pillar → Box plot statistics (Q1, Q2, Q3, IQR).',
            'representation': 'Box plot showing the distribution of E, S, G scores across all companies.',
            'x_axis': 'ESG Pillar.',
            'y_axis': 'Mean score per company.',
            'insight': 'Wide boxes indicate high variance across companies. Outlier dots show exceptional performers.',
        })
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("---")

        # Histograms for each pillar
        st.write("#### Score Histograms")
        hc1, hc2 = st.columns(2)

        with hc1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_hist_e = plots.plot_esg_histogram(date_filtered_all, "E_score")
            render_with_explanation(fig_hist_e, {
                'formula': esg_engine.FORMULAS['E_score'],
                'representation': 'Histogram showing how Environmental scores are distributed across all companies.',
                'x_axis': 'Environmental Score.',
                'y_axis': 'Number of companies.',
                'insight': 'A bell curve suggests a competitive landscape. Right-skew means most companies perform well.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

        with hc2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_hist_s = plots.plot_esg_histogram(date_filtered_all, "S_score")
            render_with_explanation(fig_hist_s, {
                'formula': esg_engine.FORMULAS['S_score'],
                'representation': 'Histogram showing how Social scores are distributed across all companies.',
                'x_axis': 'Social Score.',
                'y_axis': 'Number of companies.',
                'insight': 'Shows the overall landscape of social performance across the tracked universe.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

        hc3, hc4 = st.columns(2)

        with hc3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_hist_g = plots.plot_esg_histogram(date_filtered_all, "G_score")
            render_with_explanation(fig_hist_g, {
                'formula': esg_engine.FORMULAS['G_score'],
                'representation': 'Histogram showing how Governance scores are distributed across all companies.',
                'x_axis': 'Governance Score.',
                'y_axis': 'Number of companies.',
                'insight': 'Governance often has the widest spread due to varied corporate structures.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

        with hc4:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_hist_o = plots.plot_esg_histogram(date_filtered_all, "Overall_ESG")
            render_with_explanation(fig_hist_o, {
                'formula': esg_engine.FORMULAS['Overall_ESG'],
                'representation': 'Histogram of the composite ESG score across all companies.',
                'x_axis': 'Overall ESG Score.',
                'y_axis': 'Number of companies.',
                'insight': 'Overall distribution — the red dashed line shows the industry mean benchmark.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("---")

        # Outlier detection scatter
        st.write("#### 📌 Outlier Detection")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_out = plots.plot_outlier_scatter(date_filtered_all, company)
        render_with_explanation(fig_out, {
            'formula': 'IQR Method: Outlier if score < Q1−1.5×IQR or > Q3+1.5×IQR',
            'representation': 'Scatter plot of companies. Outliers are color-coded. Selected company marked with a star.',
            'x_axis': 'Environmental Score (company mean).',
            'y_axis': 'Governance Score (company mean).',
            'insight': 'Red dots are low-performing outliers. Green dots are high-performing outliers. Star marks your selected company.',
        })
        st.markdown('</div>', unsafe_allow_html=True)

        # Sentiment Distribution
        st.write("---")
        st.write("#### 📊 Sentiment Tone Distribution")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_sd = plots.plot_sentiment_distribution(df_company, company)
        render_with_explanation(fig_sd, {
            'formula': 'Count(Articles) grouped by AvgTone bins.',
            'representation': 'Distribution of article sentiment showing overall positive/negative coverage balance.',
            'x_axis': 'GDELT Tone Score (negative = critical, positive = favorable).',
            'y_axis': 'Frequency (number of articles).',
            'insight': 'Right-skew = consistent positive messaging. Left-skew = potential controversy.',
        })
        st.markdown('</div>', unsafe_allow_html=True)

    # ═════════ TAB 12: CORRELATIONS & RADAR ═════════
    with tabs[11]:
        st.write(f"### 🔗 {company.title()} — Correlations & ESG Radar")

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_corr = plots.plot_correlation_heatmap(df_company, company)
            render_with_explanation(fig_corr, {
                'formula': 'Pearson Correlation: r = Σ[(xi−x̄)(yi−ȳ)] / [√Σ(xi−x̄)² × √Σ(yi−ȳ)²]',
                'representation': 'Heatmap showing strength of relationships between ESG metrics and Tone.',
                'x_axis': 'Variables being compared.',
                'y_axis': 'Variables being compared.',
                'insight': 'Strong positive correlations (red) indicate co-movement. Identify which pillar drives sentiment.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_radar = plots.plot_esg_radar(df_company, ind_esg_df, company)
            render_with_explanation(fig_radar, {
                'formula': 'Mean of E, S, G sub-scores mapped to radial axes.',
                'representation': 'Radar chart comparing company vs industry across all ESG pillars.',
                'x_axis': 'Angular axes: Environmental, Social, Governance.',
                'y_axis': 'Radial distance: mean normalized score.',
                'insight': f'Strategic view of <b>{company.title()}</b>\'s strengths vs industry. Larger area = better overall profile.',
            })
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("---")

        # ESG vs Industry Overlay
        st.write("#### Company vs Industry ESG Overlay")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        combined = pd.concat([
            esg_df.assign(Type="Company"),
            ind_esg_df.assign(Type="Industry")
        ])

        chart = None
        if not combined.empty and "DATE" in combined.columns:
            chart = alt.Chart(combined).mark_line(strokeWidth=2.5, point=True).encode(
                x=alt.X('DATE:T', title='Date'),
                y=alt.Y('Score:Q', title='Score'),
                color=alt.Color('Type:N', title='Data Source', scale=alt.Scale(
                    domain=['Company', 'Industry'],
                    range=['#4F46E5', '#94A3B8']
                )),
                strokeDash=alt.StrokeDash('ESG:N', title='ESG Pillar', scale=alt.Scale(
                    domain=['E_score', 'S_score', 'G_score'],
                    range=[[0,0], [4,2], [1,1]]
                )),
                shape=alt.Shape('ESG:N', title='ESG Pilar Markers', scale=alt.Scale(
                    domain=['E_score', 'S_score', 'G_score'],
                    range=['circle', 'square', 'diamond']
                )),
                tooltip=['DATE', 'Type', 'ESG', 'Score']
            ).properties(
                title=f"{company.title()} vs Industry ESG Benchmark",
                height=350
            ).configure_view(strokeOpacity=0)

        # Passing Altair chart to render_with_explanation
        render_with_explanation(chart, {
            'formula': 'Company pillar scores vs industry-wide average per pillar.',
            'representation': 'Line overlay comparing company ESG trajectory against industry baseline.',
            'x_axis': 'Date timeline.',
            'y_axis': 'Normalized score.',
            'insight': f'Areas where <b>{company.title()}</b> line is above the grey = outperformance. Below = underperformance.',
        })
        st.markdown('</div>', unsafe_allow_html=True)

    # ═════════ TAB 13: RECOMMENDATIONS ═════════
    with tabs[12]:
        st.write(f"### 💡 AI-Driven ESG Improvement Strategy for {company.title()}")
        st.markdown("""
        <div class="insight-box">
        This recommendation engine analyzes the company's ESG scores relative to industry benchmarks
        and generates <b>data-driven, actionable improvement strategies</b> for each pillar.
        Severity is classified based on standard deviation from the industry mean.
        </div>
        """, unsafe_allow_html=True)
        st.write("")

        # Compute recommendations
        industry_bench = esg_engine.compute_industry_benchmark(date_filtered_all)

        try:
            industry_std = recommendation_engine.compute_industry_std(date_filtered_all)
        except Exception:
            industry_std = None

        recommendations = recommendation_engine.generate_recommendations(
            company_scores=company_summary,
            industry_scores=industry_bench,
            industry_std=industry_std,
        )

        render_recommendations(company, recommendations)

        # Visual summary
        st.write("---")
        st.write("#### 📊 Score vs Industry Benchmark")

        rec_cols = st.columns(3)
        pillar_data = [
            ("🌿 Environmental", "environmental", esg_engine.PILLAR_COLORS["E_score"]),
            ("👥 Social", "social", esg_engine.PILLAR_COLORS["S_score"]),
            ("🏛️ Governance", "governance", esg_engine.PILLAR_COLORS["G_score"]),
        ]

        for col_widget, (title, key, color) in zip(rec_cols, pillar_data):
            rec_entry = recommendations.get(key, {})
            with col_widget:
                severity = rec_entry.get("severity", "strong")
                border_color = {"critical": "#EF4444", "watch": "#F97316", "strong": "#10B981", "excellent": "#6366F1"}.get(severity, color)

                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.7); padding: 1.2rem; border-radius: 16px;
                            border-left: 5px solid {border_color}; margin-bottom: 0.5rem;">
                    <div style="font-weight: 700; font-size: 1rem; color: #0F172A;">{title}</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: {border_color}; margin: 0.3rem 0;">
                        {rec_entry.get('company_display', 0):.1f}
                    </div>
                    <div style="color: #64748B; font-size: 0.8rem;">
                        Industry: {rec_entry.get('industry_display', 0):.1f} |
                        Gap: <b>{rec_entry.get('gap_pts', 0):+.1f} pts</b>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.85rem;">
                        {rec_entry.get('icon', '')} {severity.capitalize()}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ═════════ TAB 14: ADVANCED GDELT ═════════
    with tabs[13]:
        st.write(f"### 🔍 Advanced GDELT Intelligence for {company.title()}")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.write("#### 👥 Key Actors (Persons)")
            if "Person" in df_company.columns:
                top_persons = df_company[df_company.Person != "UNKNOWN"]["Person"].value_counts().head(10)
                if not top_persons.empty:
                    fig_p = px.bar(top_persons, orientation='h',
                                  labels={'value': 'Mentions', 'index': 'Person'},
                                  color_discrete_sequence=['#4F46E5'])
                    fig_p.update_layout(showlegend=False, height=400, template="simple_white",
                                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_p, use_container_width=True)
                else:
                    st.info("No person data in this subset.")
            else:
                st.info("Person data not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_g2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.write("#### 📍 Top Locations")
            if "Location" in df_company.columns:
                top_locs = df_company[df_company.Location != "GLOBAL"]["Location"].value_counts().head(10)
                if not top_locs.empty:
                    fig_l = px.pie(top_locs, values=top_locs.values, names=top_locs.index,
                                  hole=0.4, color_discrete_sequence=px.colors.sequential.Blues)
                    fig_l.update_layout(height=400, template="simple_white",
                                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_l, use_container_width=True)
                else:
                    st.info("No location data in this subset.")
            else:
                st.info("Location data not available.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("---")

        # Top News Sources
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_tns = plots.plot_top_news_sources(df_company, company)
        render_with_explanation(fig_tns, {
            'formula': 'Top 10 SourceCommonName by article count.',
            'representation': 'Identifies the most active publishers covering the company.',
            'x_axis': 'Number of articles.',
            'y_axis': 'Publisher / media source.',
            'insight': f'Understanding who reports on <b>{company.title()}</b> helps gauge media reach and potential bias.',
        })
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("---")

        # Publisher Bias
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_pb = plots.plot_publisher_bias(df_company, company)
        render_with_explanation(fig_pb, {
            'formula': 'Average Tone grouped by Publishing Organization.',
            'representation': 'Horizontal bar chart revealing which publishers are positive vs negative.',
            'x_axis': 'Average sentiment tone.',
            'y_axis': 'Publisher name.',
            'insight': f'Media intelligence for <b>{company.title()}</b> — highlights potential editorial bias in specific outlets.',
        })
        st.markdown('</div>', unsafe_allow_html=True)

    # ═════════ TAB 15: 🕸️ NETWORK XAI ═════════
    with tabs[14]:
        st.write(f"### 🕸️ {company.title()} — Node2Vec Graph Explainability")
        st.markdown("""
        <div class="insight-box">
        <b>Explainable AI (XAI) for Graph Neural Networks:</b> This section decodes the Node2Vec algorithm. 
        It extracts the strongest graph connections for the target company and explains <b>why</b> they are linked by identifying the overlapping news coverage and shared ESG themes.
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        
        # 1. Get Top Connections
        df_conn = data.get("conn")
        top_conns = graph_explainability.get_top_connections(company, df_conn, top_n=5)
        
        if not top_conns:
            st.info(f"No graph embedding connections found for {company.title()} in the current dataset. Node2Vec model may not have had enough occurrences to embed this company.")
        else:
            st.write("#### 🧪 Interactive Counterfactual Analysis")
            st.markdown("Simulate 'What-If' scenarios to see how excluding certain themes affects graph connections.")
            
            c_cf1, c_cf2 = st.columns(2)
            with c_cf1:
                exclude_pillar = st.selectbox("Exclude ESG Pillar:", ["None", "Environmental", "Social", "Governance"])
            with c_cf2:
                exclude_sentiment = st.selectbox("Exclude Sentiment:", ["None", "Negative", "Positive"])
                
            ex_pillar_arg = None if exclude_pillar == "None" else exclude_pillar
            ex_sent_arg = None if exclude_sentiment == "None" else exclude_sentiment

            # 2. Compute Explanations & Counterfactuals
            explanations_dict = {}
            for conn in top_conns:
                neighbor = conn["neighbor"]
                exp = graph_explainability.explain_connection(
                    company, neighbor, date_filtered_all, 
                    original_confidence=conn["confidence"],
                    exclude_pillar=ex_pillar_arg, 
                    exclude_sentiment=ex_sent_arg
                )
                explanations_dict[neighbor.title()] = exp
                
                # Update confidence for graph visualization based on simulation
                conn["confidence"] = exp.get("simulated_confidence", conn["confidence"])
                
            # 3. Visualization (Pyvis Only)
            st.markdown('<div class="chart-container" style="padding: 0;">', unsafe_allow_html=True)
            html_graph = plots.plot_ego_network_pyvis(company, top_conns, explanations_dict)
            components.html(html_graph, height=520)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.write("---")
            st.write("#### 🕵️ Detailed Connection XAI Report")
            
            for conn in top_conns:
                neighbor = conn["neighbor"].title()
                conf = conn["confidence"]
                exp = explanations_dict[neighbor]
                
                # Determine colors based on pillar
                pillar = exp['dominant_pillar']
                p_color = "#10B981" if pillar == "Environmental" else "#3B82F6" if pillar == "Social" else "#F97316" if pillar == "Governance" else "#6366F1"
                
                with st.expander(f"🤝 Connection: {company.title()} ↔ {neighbor} (Confidence: {conf:.2f})", expanded=True):
                    st.markdown(f"**Explanation:** {exp['explanation']}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Shared Articles", f"{exp['shared_articles_count']}")
                    c2.markdown(f"**Dominant Pillar:** <span style='color:{p_color}; font-weight:bold;'>{pillar}</span>", unsafe_allow_html=True)
                    c3.metric("Average Tone", f"{exp['avg_tone']:.2f}")

    # ═════════ TAB 16: 🔬 ACADEMIC EVALUATION & BENCHMARKS ═════════
    with tabs[15]:
        st.write(f"### 🔬 Research Evaluation Suite & Paper Benchmarks ({company.title()})")
        st.write("")

        # Compute Benchmarks
        with st.spinner("Running quantitative benchmark evaluation..."):
            rating_bench = evaluation_engine.benchmark_esg_rating_alignment(date_filtered_all)
            graph_bench = evaluation_engine.benchmark_graph_node2vec_link_prediction(data.get("conn"))
            risk_bench = evaluation_engine.benchmark_risk_detection(df_company)
            xai_bench = evaluation_engine.benchmark_xai_fidelity()

        # 1. ESG Rating Alignment Benchmark
        st.write("#### 📊 Task 1: ESG Score Alignment & Benchmark Correlation")
        st.caption("Evaluates prediction correlation (Pearson r, Spearman ρ), MAE, RMSE, and Top-5 Ranking Accuracy (NDCG).")
        if rating_bench:
            df_rating_res = pd.DataFrame(rating_bench).T
            st.dataframe(df_rating_res, use_container_width=True)
        else:
            st.info("Insufficient data to run ESG rating alignment benchmark.")

        st.write("---")

        # 2. Node2Vec Graph Link Prediction Benchmark
        st.write("#### 🕸️ Task 2: Node2Vec Graph Embedding & Competitor Link Prediction")
        st.caption("Evaluates graph representation quality for competitor node embedding against standard graph baselines.")
        if graph_bench:
            df_graph_res = pd.DataFrame(graph_bench).T
            st.dataframe(df_graph_res, use_container_width=True)
        else:
            st.info("Insufficient graph connection data for link prediction evaluation.")

        st.write("---")

        # 3. Risk Sentinel Anomaly Detection
        st.write("#### ⚠️ Task 3: Risk Sentinel Anomaly Detection Accuracy")
        st.caption("Evaluates precision, recall, F1-score, and ROC-AUC in detecting extreme negative media events.")
        if risk_bench:
            df_risk_res = pd.DataFrame(risk_bench).T
            st.dataframe(df_risk_res, use_container_width=True)
        else:
            st.info("Insufficient time-series data for risk anomaly detection evaluation.")

        st.write("---")

        # 4. XAI Faithfulness & Counterfactual Evaluation
        st.write("#### 🧪 Task 4: Explainable AI (XAI) & Counterfactual Faithfulness")
        st.caption("Quantifies explanation fidelity, counterfactual validity, and human inter-rater agreement.")
        cols_xai = st.columns(2)
        for idx, (k, v) in enumerate(xai_bench.items()):
            cols_xai[idx % 2].metric(k, str(v))

        st.write("---")


    # ═════════ TAB 17: 📄 DOWNLOAD REPORT ═════════
    with tabs[16]:
        st.write(f"### 📄 {company.title()} — Professional ESG Intelligence Report")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        st.write("#### Generate Professional Performance Report")
        st.write("This PDF contains a comprehensive AI-powered breakdown of metrics, risks, and forecasts.")
        
        # Prepare Data for Report
        exec_metrics = esg_engine.compute_executive_metrics(df_company, data["data"])
        pillar_scores = esg_engine.compute_company_summary(df_company, df_global_scored)
        
        df_hist_w = esg_engine.compute_esg_aggregates(df_company, 'weekly')
        df_pred_w = prediction_engine.predict_future_scores(df_hist_w)
        forecast_summ = prediction_engine.get_forecast_summary(df_hist_w, df_pred_w) if not df_pred_w.empty else (0, "stable")
        
        risks_report = []
        for col in ["Overall_ESG", "E_score", "S_score", "G_score"]:
            risks_report.extend(risk_engine.detect_risks(df_company, column=col))
            
        story_report = narrative_engine.generate_performance_story(esg_engine.compute_esg_aggregates(df_company, 'monthly'))

        if st.button("🚀 Prepare AI Intelligence Report"):
            with st.spinner("Compiling comprehensive intelligence from all modules..."):
                # Gather extra intelligence
                # 1. Recommendations
                industry_bench = esg_engine.compute_industry_benchmark(date_filtered_all)
                try:
                    industry_std = recommendation_engine.compute_industry_std(date_filtered_all)
                except Exception:
                    industry_std = None
                
                recs_dict = recommendation_engine.generate_recommendations(
                    company_scores=company_summary,
                    industry_scores=industry_bench,
                    industry_std=industry_std,
                )
                
                recs = []
                for p in ["environmental", "social", "governance", "overall"]:
                    if p in recs_dict and "recommendations" in recs_dict[p]:
                        recs.extend(recs_dict[p]["recommendations"])
                
                # 2. Score Breakdown
                factors_list = explainability_engine.get_factor_percentages("Overall_ESG", scores=company_summary)
                factors = {}
                for f in factors_list:
                    factors[f[0]] = {
                        "primary_factor": f"{f[1]}% weighting",
                        "explanation": f"Derived from the relative amplitude of the {f[0]} signals compared to other pillars."
                    }
                
                # 3. Rankings
                rank = df_global_scored[df_global_scored['Organization'].str.lower() == company.lower()]['Overall_ESG_Rank'].min() if 'Overall_ESG_Rank' in df_global_scored else "N/A"
                total_ranked = len(df_global_scored['Organization'].unique()) if 'Organization' in df_global_scored else "N/A"
                
                # 4. Correlations
                # Select only the numeric columns needed for correlation to avoid errors
                numeric_df = df_company[['E_score', 'S_score', 'G_score']].select_dtypes(include=[np.number])
                corrs = numeric_df.corr().round(2).to_dict() if not numeric_df.empty else {}
                
                # 5. Advanced GDELT
                top_persons = df_company[df_company.Person != "UNKNOWN"]["Person"].value_counts().head(5).to_dict() if "Person" in df_company.columns else {}
                top_locs = df_company[df_company.Location != "GLOBAL"]["Location"].value_counts().head(5).to_dict() if "Location" in df_company.columns else {}
                top_pubs = df_company["SourceCommonName"].value_counts().head(5).to_dict() if "SourceCommonName" in df_company.columns else {}
                
                # 6. Network XAI
                top_conns = graph_explainability.get_top_connections(company, data.get("conn"), top_n=3)
                network_xai = []
                for conn in top_conns:
                    exp = graph_explainability.explain_connection(company, conn["neighbor"], date_filtered_all)
                    network_xai.append({
                        "neighbor": conn["neighbor"].title(),
                        "confidence": conn["confidence"],
                        "explanation": exp["explanation"]
                    })
                
                # 7. Individual Articles (Tab 2)
                def _get_top_articles(df_sub):
                    recs = []
                    for _, r in df_sub.iterrows():
                        e = r.get("E", False)
                        s = r.get("S", False)
                        g = r.get("G", False)
                        pillar = "Environmental" if e else ("Social" if s else ("Governance" if g else "General ESG"))
                        recs.append({
                            "URL": str(r.get("URL", "#")),
                            "Tone": float(r.get("Tone", 0.0)),
                            "Dominant_Pillar": pillar
                        })
                    return recs

                if 'Tone' in df_company.columns and not df_company.empty:
                    top_pos = _get_top_articles(df_company.nlargest(3, 'Tone'))
                    top_neg = _get_top_articles(df_company.nsmallest(3, 'Tone'))
                else:
                    top_pos, top_neg = [], []
                
                # 8. Trends & Time Series (Tabs 3, 4, 5)
                df_hist_m = esg_engine.compute_esg_aggregates(df_company, 'monthly')
                monthly_table = df_hist_m.tail(6).to_dict('records') if not df_hist_m.empty else []
                weekly_table = df_hist_w.tail(6).to_dict('records') if not df_hist_w.empty else []
                
                trend_slopes = {
                    "E": float(esg_engine.compute_momentum(df_hist_w['E_score'])) if not df_hist_w.empty and 'E_score' in df_hist_w.columns else 0.0,
                    "S": float(esg_engine.compute_momentum(df_hist_w['S_score'])) if not df_hist_w.empty and 'S_score' in df_hist_w.columns else 0.0,
                    "G": float(esg_engine.compute_momentum(df_hist_w['G_score'])) if not df_hist_w.empty and 'G_score' in df_hist_w.columns else 0.0
                }
                
                # 9. Peer Comparison (Tab 10)
                ranked_df = esg_engine.rank_companies(date_filtered_all)
                if not ranked_df.empty:
                    comp_row = ranked_df[ranked_df['Organization'].str.lower() == company.lower()]
                    rank = int(comp_row['Rank'].iloc[0]) if not comp_row.empty else "N/A"
                    total_ranked = len(ranked_df)
                    peer_comp = ranked_df.head(5).rename(columns={
                        'Overall_mean': 'Overall_ESG', 
                        'E_mean': 'E_score', 
                        'S_mean': 'S_score', 
                        'G_mean': 'G_score'
                    }).to_dict('records')
                else:
                    rank, total_ranked = "N/A", "N/A"
                    peer_comp = []
                
                # 10. Distributions (Tab 11)
                num_cols = ["Tone", "Overall_ESG", "E_score", "S_score", "G_score"]
                avail_cols = [c for c in num_cols if c in df_company.columns]
                stats_dist = df_company[avail_cols].describe().round(3).to_dict() if avail_cols else {}

                report_data = {
                    "company_name": company,
                    "exec_metrics": exec_metrics,
                    "pillar_scores": pillar_scores,
                    "forecast_summary": forecast_summ,
                    "risks": risks_report,
                    "story": story_report,
                    "recommendations": recs,
                    "factors": factors,
                    "rank": rank,
                    "total_ranked": total_ranked,
                    "correlations": corrs,
                    "top_persons": top_persons,
                    "top_locations": top_locs,
                    "top_publishers": top_pubs,
                    "network_xai": network_xai,
                    "top_pos_articles": top_pos,
                    "top_neg_articles": top_neg,
                    "monthly_table": monthly_table,
                    "weekly_table": weekly_table,
                    "trend_slopes": trend_slopes,
                    "peer_comp": peer_comp,
                    "stats_dist": stats_dist
                }
                
                pdf_buffer = report_generator.generate_esg_pdf(report_data)
                
                st.success("Report Generated Successfully!")
                st.download_button(
                    label="📥 Download ESG Report (PDF)",
                    data=pdf_buffer,
                    file_name=f"ESG_Report_{company.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        start_data = "jan1"
        end_data = "jan5"
    else:
        start_data = args[1]
        end_data = args[2]

    if f"{start_data}_to_{end_data}" not in os.listdir("Data"):
        st.error(f"There isn't data for {start_data}_to_{end_data}. Please pick from {os.listdir('Data')}")
        st.stop()
    else:
        main(start_data, end_data)
    alt.themes.enable("default")
