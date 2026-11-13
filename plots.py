"""
plots.py
========
Professional Plotly visualizations for the ESG Analytics Dashboard.
All scoring logic is imported from esg_engine.py (single source of truth).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from esg_engine import (
    normalize,
    ensure_columns,
    compute_pillar_scores,
    preprocess_company_data,
    rank_companies,
    detect_outliers,
    PILLAR_COLORS,
    PILLAR_LABELS,
    PILLAR_WEIGHTS,
    FORMULAS,
)


# ═══════════════════════════════════════════════════════════════════
#  THEME CONSTANTS
# ═══════════════════════════════════════════════════════════════════

ESG_COLORS = {
    "E": PILLAR_COLORS["E_score"],
    "S": PILLAR_COLORS["S_score"],
    "G": PILLAR_COLORS["G_score"],
    "Overall": PILLAR_COLORS["Overall_ESG"],
    "Neutral": "#94A3B8",
}

PLOTLY_TEMPLATE = "plotly_white"


def _hex_to_rgba(hex_color: str, opacity: float) -> str:
    """Convert #RRGGBB to rgba(r,g,b,a)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {opacity})"


def apply_custom_style(fig, title: str):
    """Apply premium Plotly styling to any figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20, "family": "Outfit, sans-serif"},
        },
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#334155"),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.05)",
            borderwidth=1,
            font=dict(size=12),
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
#  1. INDIVIDUAL PILLAR SCORE (Area Chart)
# ═══════════════════════════════════════════════════════════════════

def plot_individual_pillar(df_company, company_name, pillar_type):
    """Separate area graph for a single ESG pillar or Overall."""
    if df_company.empty:
        return None

    pillar_map = {
        "Environmental": ("E_score", ESG_COLORS["E"]),
        "Social": ("S_score", ESG_COLORS["S"]),
        "Governance": ("G_score", ESG_COLORS["G"]),
        "Overall": ("Overall_ESG", ESG_COLORS["Overall"]),
    }

    if pillar_type not in pillar_map:
        return None

    col, color = pillar_map[pillar_type]
    if col not in df_company.columns:
        return None

    sorted_df = df_company.sort_values("DATE")

    fig = px.area(
        sorted_df,
        x="DATE",
        y=col,
        color_discrete_sequence=[color],
    )

    # Add Mean Line
    mean_val = sorted_df[col].mean()
    fig.add_hline(
        y=mean_val,
        line_dash="dash",
        line_color=color,
        annotation_text=f"Average: {mean_val:.3f}",
        annotation_position="bottom right",
    )

    fig = apply_custom_style(fig, f"{pillar_type} Score: {company_name}")
    fig.update_traces(fillcolor=_hex_to_rgba(color, 0.15), line=dict(width=3))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=f"{pillar_type} Score (Z-normalized)")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  2. ESG PILLAR TRENDS (Combined Line)
# ═══════════════════════════════════════════════════════════════════

def plot_esg_trend(df_company, company_name):
    """Line chart showing E, S, G scores over time."""
    if df_company.empty:
        return None

    # Aggregate by date
    score_cols = ["E_score", "S_score", "G_score"]
    existing = [c for c in score_cols if c in df_company.columns]
    if not existing:
        return None

    df_daily = df_company.groupby("DATE")[existing].mean().reset_index()

    fig = go.Figure()

    pillars = [
        ("E_score", "Environmental", ESG_COLORS["E"]),
        ("S_score", "Social", ESG_COLORS["S"]),
        ("G_score", "Governance", ESG_COLORS["G"]),
    ]

    for col_name, label, color in pillars:
        if col_name not in df_daily.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df_daily["DATE"],
                y=df_daily[col_name],
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=3),
                marker=dict(size=5),
            )
        )

    fig = apply_custom_style(fig, f"ESG Pillar Trends: {company_name}")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Score (Z-normalized)")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  3. ESG PILLAR COMPARISON (Bar Chart)
# ═══════════════════════════════════════════════════════════════════

def plot_esg_pillar_comparison(df_company, company_name):
    """Bar chart comparing aggregated E, S, G scores."""
    if df_company.empty:
        return None

    score_cols = {"Environmental": "E_score", "Social": "S_score", "Governance": "G_score"}
    avg_scores = {}
    for label, col in score_cols.items():
        if col in df_company.columns:
            avg_scores[label] = df_company[col].mean()

    if not avg_scores:
        return None

    plot_df = pd.DataFrame(list(avg_scores.items()), columns=["Pillar", "Score"])

    fig = px.bar(
        plot_df,
        x="Pillar",
        y="Score",
        color="Pillar",
        color_discrete_map={
            "Environmental": ESG_COLORS["E"],
            "Social": ESG_COLORS["S"],
            "Governance": ESG_COLORS["G"],
        },
        text_auto=".3f",
    )

    fig = apply_custom_style(fig, f"ESG Pillar Comparison: {company_name}")
    fig.update_traces(marker_line_width=0, opacity=0.88)
    fig.update_xaxes(title_text="ESG Pillar")
    fig.update_yaxes(title_text="Mean Normalized Score")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  4. SENTIMENT DISTRIBUTION (Histogram + Box)
# ═══════════════════════════════════════════════════════════════════

def plot_sentiment_distribution(df_company, company_name):
    """Histogram with mean/median annotations and marginal boxplot."""
    if df_company.empty:
        return None
    tone_col = "AvgTone" if "AvgTone" in df_company.columns else "Tone"
    if tone_col not in df_company.columns:
        return None

    mean_val = df_company[tone_col].mean()

    fig = px.histogram(
        df_company,
        x=tone_col,
        nbins=30,
        color_discrete_sequence=[ESG_COLORS["S"]],
        marginal="box",
    )

    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top left",
    )

    fig = apply_custom_style(fig, f"Sentiment Distribution: {company_name}")
    fig.update_xaxes(title_text="GDELT Tone Score")
    fig.update_yaxes(title_text="Frequency (Article Count)")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  5. SENTIMENT MOMENTUM (Area Chart)
# ═══════════════════════════════════════════════════════════════════

def plot_sentiment_momentum(df_tone, company_name):
    """Area chart for sentiment momentum over time."""
    if df_tone.empty:
        return None

    fig = px.area(df_tone, x="DATE", y="Score")
    fig = apply_custom_style(fig, f"Sentiment Momentum: {company_name}")
    fig.update_traces(
        fillcolor="rgba(79, 70, 229, 0.15)",
        line=dict(color="#4F46E5", width=3),
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Sentiment Score")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  6. TOP NEWS SOURCES (Horizontal Bar)
# ═══════════════════════════════════════════════════════════════════

def plot_top_news_sources(df_company, company_name):
    """Bar chart for top 10 publishers covering the company."""
    if df_company.empty or "SourceCommonName" not in df_company.columns:
        return None

    pub_counts = df_company["SourceCommonName"].value_counts().head(10).reset_index()
    pub_counts.columns = ["Publisher", "Count"]

    fig = px.bar(
        pub_counts,
        x="Count",
        y="Publisher",
        orientation="h",
        color_discrete_sequence=["#06B6D4"],
    )

    fig = apply_custom_style(fig, f"Top 10 News Sources: {company_name}")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.update_xaxes(title_text="Number of Articles")
    fig.update_yaxes(title_text="Publisher")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  7. CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(df_company, company_name):
    """Pearson correlation heatmap between ESG metrics and Tone."""
    if df_company.empty:
        return None

    cols = ["AvgTone", "NumArticles", "NumMentions", "E_score", "S_score", "G_score", "Overall_ESG"]
    existing_cols = [c for c in cols if c in df_company.columns]

    if len(existing_cols) < 2:
        return None

    corr_matrix = df_company[existing_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )

    fig = apply_custom_style(fig, f"ESG & Sentiment Correlation: {company_name}")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  8. COMPANY RANKING (Horizontal Bar with highlight)
# ═══════════════════════════════════════════════════════════════════

def plot_company_ranking(df_all, current_company, top_n=20):
    """
    Horizontal bar chart ranking companies by Overall ESG score.
    Highlights the selected company.
    """
    if df_all.empty:
        return None

    ranked = rank_companies(df_all)
    if ranked.empty:
        return None

    # Show top N
    top = ranked.head(top_n).copy()

    top["Color"] = top["Organization"].apply(
        lambda x: ESG_COLORS["Overall"]
        if x.lower() == current_company.lower()
        else ESG_COLORS["Neutral"]
    )

    top["Label"] = top.apply(
        lambda r: f"#{int(r['Rank'])} {r['Organization'].title()}", axis=1
    )

    fig = px.bar(
        top,
        x="Display_ESG",
        y="Label",
        orientation="h",
        color="Color",
        color_discrete_map="identity",
        text=top["Display_ESG"].apply(lambda x: f"{x:.1f}"),
    )

    fig = apply_custom_style(fig, "Company ESG Ranking (Top 20)")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
        height=max(500, top_n * 30),
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(title_text="ESG Score (0-100 Scale)", range=[0, 105])
    fig.update_yaxes(title_text="")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  9. SENTIMENT VOLATILITY (Rolling Std)
# ═══════════════════════════════════════════════════════════════════

def plot_sentiment_volatility(df_company, company_name):
    """Line chart showing rolling sentiment volatility."""
    if df_company.empty:
        return None

    tone_col = "AvgTone" if "AvgTone" in df_company.columns else "Tone"
    if tone_col not in df_company.columns:
        return None

    df_vol = df_company.sort_values("DATE").copy()
    n_points = len(df_vol)
    window = max(3, min(14, n_points // 10 + 3))

    df_vol["Volatility"] = normalize(df_vol[tone_col]).rolling(window).std()

    fig = px.line(df_vol, x="DATE", y="Volatility")
    fig = apply_custom_style(fig, f"Sentiment Volatility Index: {company_name}")
    fig.update_traces(line=dict(color="#EF4444", width=3))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Volatility (Rolling Std)")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  10. ESG RADAR (Company vs Industry)
# ═══════════════════════════════════════════════════════════════════

def plot_esg_radar(df_company, df_industry, company_name):
    """Radar chart comparing company vs industry averages across pillars."""
    if df_company.empty:
        return None

    pillars = ["E_score", "S_score", "G_score"]
    labels = ["Environmental", "Social", "Governance"]

    company_vals = []
    for p in pillars:
        company_vals.append(df_company[p].mean() if p in df_company.columns else 0)

    # Industry values
    if isinstance(df_industry, pd.DataFrame) and not df_industry.empty:
        if "Score" in df_industry.columns and "ESG" in df_industry.columns:
            # Melted format
            ind_means = df_industry.groupby("ESG")["Score"].mean()
            industry_vals = [ind_means.get(p, 0) for p in pillars]
        elif all(p in df_industry.columns for p in pillars):
            industry_vals = [df_industry[p].mean() for p in pillars]
        else:
            industry_vals = [0, 0, 0]
    else:
        industry_vals = [0, 0, 0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=company_vals + [company_vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=company_name,
            line=dict(color=ESG_COLORS["Overall"], width=3),
            fillcolor=_hex_to_rgba(ESG_COLORS["Overall"], 0.15),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=industry_vals + [industry_vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Industry Avg",
            line=dict(color=ESG_COLORS["Neutral"], dash="dash", width=2),
            fillcolor=_hex_to_rgba(ESG_COLORS["Neutral"], 0.08),
        )
    )

    fig = apply_custom_style(fig, f"{company_name} vs Industry Profile")
    max_r = max(max(company_vals), max(industry_vals), 0.1) * 1.2
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-max_r, max_r])),
        height=450,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
#  11. PUBLISHER SENTIMENT BIAS
# ═══════════════════════════════════════════════════════════════════

def plot_publisher_bias(df_company, company_name):
    """Bar chart showing mean sentiment per publisher."""
    if df_company.empty or "SourceCommonName" not in df_company.columns:
        return None

    tone_col = "AvgTone" if "AvgTone" in df_company.columns else "Tone"
    if tone_col not in df_company.columns:
        return None

    pub_sentiment = df_company.groupby("SourceCommonName")[tone_col].mean().sort_values()
    plot_df = pub_sentiment.reset_index()
    plot_df.columns = ["Publisher", "Sentiment"]

    fig = px.bar(
        plot_df,
        x="Sentiment",
        y="Publisher",
        orientation="h",
        color="Sentiment",
        color_continuous_scale="RdBu_r",
    )

    fig = apply_custom_style(fig, f"Publisher Sentiment Bias: {company_name}")
    fig.update_xaxes(title_text="Average Tone Score")
    fig.update_yaxes(title_text="Publisher")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  12. BEST & WORST COMPANIES (Dual Bar)
# ═══════════════════════════════════════════════════════════════════

def plot_best_worst_companies(df_all, n=5):
    """Highlights the top-N best and bottom-N worst companies."""
    if df_all.empty:
        return None

    ranked = rank_companies(df_all)
    if len(ranked) < 2:
        return None

    best = ranked.head(n).copy()
    worst = ranked.tail(n).copy()

    best["Category"] = "🏆 Top Performers"
    worst["Category"] = "⚠️ Needs Improvement"

    combined = pd.concat([best, worst])
    combined["Org_Label"] = combined["Organization"].str.title()

    fig = px.bar(
        combined,
        x="Display_ESG",
        y="Org_Label",
        color="Category",
        orientation="h",
        color_discrete_map={
            "🏆 Top Performers": "#10B981",
            "⚠️ Needs Improvement": "#EF4444",
        },
        text=combined["Display_ESG"].apply(lambda x: f"{x:.1f}"),
        barmode="group",
    )

    fig = apply_custom_style(fig, f"Best & Worst Performing Companies")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=max(400, n * 60),
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(title_text="ESG Score (0-100 Scale)", range=[0, 105])
    fig.update_yaxes(title_text="")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  13. ESG DISTRIBUTION BOX PLOT
# ═══════════════════════════════════════════════════════════════════

def plot_esg_distribution_boxplot(df_all):
    """Box plot showing distribution of E, S, G scores across all companies."""
    if df_all.empty:
        return None

    df_all = ensure_columns(df_all)
    df_all = compute_pillar_scores(df_all)

    # Compute per-company means
    company_means = (
        df_all.groupby("Organization")
        .agg(Environmental=("E_score", "mean"), Social=("S_score", "mean"), Governance=("G_score", "mean"))
        .reset_index()
    )

    melted = company_means.melt(
        id_vars="Organization", var_name="Pillar", value_name="Score"
    )

    fig = px.box(
        melted,
        x="Pillar",
        y="Score",
        color="Pillar",
        color_discrete_map={
            "Environmental": ESG_COLORS["E"],
            "Social": ESG_COLORS["S"],
            "Governance": ESG_COLORS["G"],
        },
        points="all",
    )

    fig = apply_custom_style(fig, "ESG Score Distribution Across Companies")
    fig.update_xaxes(title_text="ESG Pillar")
    fig.update_yaxes(title_text="Mean Score per Company")
    fig.update_traces(marker=dict(size=5, opacity=0.5))
    return fig


# ═══════════════════════════════════════════════════════════════════
#  14. ESG HISTOGRAM PER PILLAR
# ═══════════════════════════════════════════════════════════════════

def plot_esg_histogram(df_all, pillar="Overall_ESG"):
    """Histogram of a single pillar across all companies."""
    if df_all.empty:
        return None

    df_all = ensure_columns(df_all)
    df_all = compute_pillar_scores(df_all)

    company_means = df_all.groupby("Organization")[pillar].mean().reset_index()

    label = PILLAR_LABELS.get(pillar, pillar)
    color = PILLAR_COLORS.get(pillar, "#6366F1")

    fig = px.histogram(
        company_means,
        x=pillar,
        nbins=20,
        color_discrete_sequence=[color],
        marginal="rug",
    )

    mean_val = company_means[pillar].mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Mean: {mean_val:.3f}",
    )

    fig = apply_custom_style(fig, f"{label} Score Distribution")
    fig.update_xaxes(title_text=f"{label} Score")
    fig.update_yaxes(title_text="Number of Companies")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  15. WEIGHTED ESG WATERFALL
# ═══════════════════════════════════════════════════════════════════

def plot_weighted_esg_waterfall(company_scores: dict, company_name: str):
    """
    Waterfall chart showing weighted contribution of each pillar
    to the total ESG score.
    """
    e_contrib = company_scores.get("e_mean", 0) * PILLAR_WEIGHTS["E_score"]
    s_contrib = company_scores.get("s_mean", 0) * PILLAR_WEIGHTS["S_score"]
    g_contrib = company_scores.get("g_mean", 0) * PILLAR_WEIGHTS["G_score"]
    total = e_contrib + s_contrib + g_contrib

    fig = go.Figure(
        go.Waterfall(
            name="ESG Contribution",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=[
                f"Environmental\n(×{PILLAR_WEIGHTS['E_score']})",
                f"Social\n(×{PILLAR_WEIGHTS['S_score']})",
                f"Governance\n(×{PILLAR_WEIGHTS['G_score']})",
                "Total ESG",
            ],
            y=[e_contrib, s_contrib, g_contrib, total],
            text=[f"{e_contrib:+.3f}", f"{s_contrib:+.3f}", f"{g_contrib:+.3f}", f"{total:.3f}"],
            textposition="outside",
            connector={"line": {"color": "rgba(0,0,0,0.1)"}},
            increasing={"marker": {"color": "#10B981"}},
            decreasing={"marker": {"color": "#EF4444"}},
            totals={"marker": {"color": ESG_COLORS["Overall"]}},
        )
    )

    fig = apply_custom_style(fig, f"Weighted ESG Breakdown: {company_name}")
    fig.update_xaxes(title_text="Component")
    fig.update_yaxes(title_text="Weighted Score Contribution")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  16. OUTLIER DETECTION SCATTER
# ═══════════════════════════════════════════════════════════════════

def plot_outlier_scatter(df_all, current_company=None):
    """Scatter plot of companies with outliers flagged."""
    if df_all.empty:
        return None

    ranked = rank_companies(df_all)
    ranked = detect_outliers(ranked, "Overall_mean")

    fig = px.scatter(
        ranked,
        x="E_mean",
        y="G_mean",
        size="ArticleCount",
        color="outlier_type",
        hover_name="Organization",
        hover_data={"Overall_mean": ":.3f", "Rank": True},
        color_discrete_map={
            "Normal": ESG_COLORS["Neutral"],
            "High Outlier": "#10B981",
            "Low Outlier": "#EF4444",
        },
        size_max=25,
    )

    # Highlight current company
    if current_company:
        comp_row = ranked[ranked["Organization"].str.lower() == current_company.lower()]
        if not comp_row.empty:
            fig.add_trace(
                go.Scatter(
                    x=comp_row["E_mean"],
                    y=comp_row["G_mean"],
                    mode="markers+text",
                    marker=dict(size=18, color=ESG_COLORS["Overall"], symbol="star"),
                    text=[current_company.title()],
                    textposition="top center",
                    name="Selected",
                    showlegend=True,
                )
            )

    fig = apply_custom_style(fig, "Outlier Detection: ESG Landscape")
    fig.update_xaxes(title_text="Environmental Score (Mean)")
    fig.update_yaxes(title_text="Governance Score (Mean)")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  17. TREND ANALYSIS WITH MOVING AVERAGE
# ═══════════════════════════════════════════════════════════════════

def plot_trend_analysis(df_company, company_name):
    """Overall ESG trend with 7-day moving average overlay."""
    if df_company.empty or "Overall_ESG" not in df_company.columns:
        return None

    df_sorted = df_company.sort_values("DATE").copy()
    df_daily = df_sorted.groupby("DATE")["Overall_ESG"].mean().reset_index()

    window = max(3, min(7, len(df_daily) // 5))
    df_daily["MA"] = df_daily["Overall_ESG"].rolling(window, min_periods=1).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_daily["DATE"],
            y=df_daily["Overall_ESG"],
            name="Daily ESG",
            mode="lines",
            line=dict(color=_hex_to_rgba(ESG_COLORS["Overall"], 0.4), width=1.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_daily["DATE"],
            y=df_daily["MA"],
            name=f"{window}-Day Moving Avg",
            mode="lines",
            line=dict(color=ESG_COLORS["Overall"], width=3),
        )
    )

    fig = apply_custom_style(fig, f"ESG Trend Analysis: {company_name}")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Overall ESG Score")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  18. MULTI-COMPANY COMPARISON
# ═══════════════════════════════════════════════════════════════════

def plot_multi_company_comparison(df_all, selected_companies):
    """Grouped bar chart comparing E, S, G across multiple companies."""
    if df_all.empty or not selected_companies:
        return None

    df_all = ensure_columns(df_all)
    df_all = compute_pillar_scores(df_all)

    filtered = df_all[df_all["Organization"].str.lower().isin([c.lower() for c in selected_companies])]
    if filtered.empty:
        return None

    company_means = (
        filtered.groupby("Organization")
        .agg(Environmental=("E_score", "mean"), Social=("S_score", "mean"), Governance=("G_score", "mean"))
        .reset_index()
    )
    company_means["Organization"] = company_means["Organization"].str.title()

    melted = company_means.melt(id_vars="Organization", var_name="Pillar", value_name="Score")

    fig = px.bar(
        melted,
        x="Pillar",
        y="Score",
        color="Organization",
        barmode="group",
        text_auto=".3f",
    )

    fig = apply_custom_style(fig, "Multi-Company ESG Comparison")
    fig.update_xaxes(title_text="ESG Pillar")
    fig.update_yaxes(title_text="Mean Score")
    fig.update_traces(textposition="outside")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  19. TIME-BASED TREND (Monthly/Weekly Line)
# ═══════════════════════════════════════════════════════════════════

def plot_time_based_trend(df_resampled, company_name, freq_label="Month"):
    """
    Line chart showing E, S, G, and Overall ESG scores over resampled periods.
    """
    if df_resampled.empty:
        return None

    fig = go.Figure()

    pillars = [
        ("E_score", "Environmental", ESG_COLORS["E"]),
        ("S_score", "Social", ESG_COLORS["S"]),
        ("G_score", "Governance", ESG_COLORS["G"]),
        ("Overall_ESG", "Overall ESG", ESG_COLORS["Overall"]),
    ]

    for col_name, label, color in pillars:
        if col_name not in df_resampled.columns:
            continue
        
        # Convert to 0-100 display score for the plot
        display_y = df_resampled[col_name].apply(lambda x: float(np.clip(50 + x * 15, 0, 100)))
        
        fig.add_trace(
            go.Scatter(
                x=df_resampled["DATE"],
                y=display_y,
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=4 if label == "Overall ESG" else 2),
                marker=dict(size=8),
                hovertemplate=f"<b>{label}</b><br>{freq_label}: %{{x|%Y-%m-%d}}<br>Score: %{{y:.1f}}<extra></extra>"
            )
        )

    fig = apply_custom_style(fig, f"{freq_label}-wise ESG Trends: {company_name}")
    fig.update_xaxes(title_text=freq_label)
    fig.update_yaxes(title_text="ESG Score (0-100 Scale)", range=[0, 105])
    return fig


# ═══════════════════════════════════════════════════════════════════
#  20. PERIOD COMPARISON (Monthly/Weekly Bar)
# ═══════════════════════════════════════════════════════════════════

def plot_period_comparison(df_resampled, company_name, freq_label="Month"):
    """
    Bar chart comparing Overall ESG scores across periods.
    """
    if df_resampled.empty:
        return None

    # Format dates for better display on bars
    df_plot = df_resampled.copy()
    if freq_label == "Month":
        df_plot["Period"] = df_plot["DATE"].dt.strftime("%Y-%b")
    else:
        iso_year = df_plot["DATE"].dt.isocalendar().year
        iso_week = df_plot["DATE"].dt.isocalendar().week
        df_plot["Period"] = "W" + iso_week.astype(str).str.zfill(2) + "-" + iso_year.astype(str)

    # Convert to 0-100 display score
    df_plot["Display_Score"] = df_plot["Overall_ESG"].apply(lambda x: float(np.clip(50 + x * 15, 0, 100)))

    fig = px.bar(
        df_plot,
        x="Period",
        y="Display_Score",
        text_auto=".1f",
        color="Display_Score",
        color_continuous_scale="RdYlGn",
        range_y=[0, 105],
        labels={"Display_Score": "Overall ESG Score", "Period": freq_label}
    )

    fig = apply_custom_style(fig, f"{freq_label}-on-{freq_label} Comparison: {company_name}")
    fig.update_traces(marker_line_width=0, opacity=0.85)
    fig.update_layout(coloraxis_showscale=False)
    return fig


# ═══════════════════════════════════════════════════════════════════
#  21. FOCUSED PERIOD VARIATION (Daily Detail)
# ═══════════════════════════════════════════════════════════════════

def plot_focused_variation(df_period, company_name, period_label, pillar_type="Overall"):
    """
    Renders daily variation within a specific month or week.
    Highlights Peak, Lowest, and Average Line.
    """
    if df_period.empty:
        return None

    pillar_map = {
        "Environmental": ("E_score", ESG_COLORS["E"]),
        "Social": ("S_score", ESG_COLORS["S"]),
        "Governance": ("G_score", ESG_COLORS["G"]),
        "Overall": ("Overall_ESG", ESG_COLORS["Overall"]),
    }

    if pillar_type not in pillar_map:
        pillar_type = "Overall"

    col_name, color = pillar_map[pillar_type]
    if col_name not in df_period.columns:
        return None

    # Ensure data is aggregated daily
    df_plot = df_period.copy()
    org_name = df_plot['Organization'].iloc[0].title() if 'Organization' in df_plot.columns and not df_plot.empty else company_name

    # Step 1: Group by day first (average raw Z-scores per day before any scaling)
    df_plot['DateOnly'] = pd.to_datetime(df_plot['DATE']).dt.date
    daily_df = df_plot.groupby('DateOnly', as_index=False)[col_name].mean().sort_values('DateOnly')

    if daily_df.empty:
        return None

    # Step 2: Keep a copy of raw (pre-display) scores for accurate Peak/Lowest labels
    raw_scores = daily_df[col_name].copy()

    # Step 3: Convert raw Z-scores to 0-100 display scale
    daily_df[col_name] = daily_df[col_name].apply(lambda x: float(np.clip(50 + x * 15, 0, 100)))
    display_scores = daily_df[col_name].copy()

    # Step 4: Identify peak and lowest by the raw scores (before display compression),
    # so the markers always reflect the true best/worst days even when display values
    # appear close together after the global Z-score → 0-100 mapping.
    peak_idx = raw_scores.idxmax()
    lowest_idx = raw_scores.idxmin()
    peak_row = daily_df.loc[peak_idx]
    lowest_row = daily_df.loc[lowest_idx]

    # Calculate average from display scores
    avg_val = display_scores.mean()

    fig = go.Figure()

    # Main Line
    fig.add_trace(
        go.Scatter(
            x=daily_df["DateOnly"],
            y=daily_df[col_name],
            name=f"{pillar_type} Score",
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=6, color=color, opacity=0.7),
            hovertemplate=f"<b>{org_name}</b><br>Date: %{{x}}<br>Score: %{{y:.3f}}<extra></extra>"
        )
    )

    # Average Line
    fig.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color=color,
        annotation_text=f"Avg: {avg_val:.3f}",
        annotation_position="bottom right",
        opacity=0.6
    )

    # Always show Peak marker
    fig.add_trace(
        go.Scatter(
            x=[peak_row["DateOnly"]],
            y=[peak_row[col_name]],
            mode="markers+text",
            name="Peak Day",
            marker=dict(size=12, color="#10B981", symbol="triangle-up"),
            text=["Peak Day"],
            textposition="top center",
            hovertemplate=f"Peak Score: {peak_row[col_name]:.3f}<br>Date: %{{x}}<extra></extra>"
        )
    )

    # Always show Lowest marker — offset y slightly below if it coincides with Peak
    # (only happens when there is truly one data point for the period)
    lowest_y = lowest_row[col_name]
    lowest_text_pos = "bottom center"
    if peak_idx == lowest_idx:
        # Single day: nudge the Lowest marker down so both are visible
        lowest_y = max(0, lowest_y - 5)
        lowest_text_pos = "bottom center"

    fig.add_trace(
        go.Scatter(
            x=[lowest_row["DateOnly"]],
            y=[lowest_y],
            mode="markers+text",
            name="Lowest Day",
            marker=dict(size=12, color="#EF4444", symbol="triangle-down"),
            text=["Lowest Day"],
            textposition=lowest_text_pos,
            hovertemplate=f"Lowest Score: {lowest_row[col_name]:.3f}<br>Date: %{{x}}<extra></extra>"
        )
    )

    fig = apply_custom_style(fig, f"{pillar_type} Variation: {company_name} ({period_label})")
    fig.update_xaxes(title_text="Day of Period")
    fig.update_yaxes(title_text=f"{pillar_type} Score (0-100)", range=[0, 105])
    fig.update_layout(showlegend=True)

    # If peak and lowest are the same day, add a visible note on the chart
    if peak_idx == lowest_idx:
        fig.add_annotation(
            text="⚠️ Only 1 day of data available for this week.<br>Peak and Lowest refer to the same day.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12, color="#888888"),
            align="center",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=6,
        )

    return fig


# ═══════════════════════════════════════════════════════════════════
#  22. GRAPH XAI NETWORK VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def plot_ego_network_plotly(company_name, connections):
    """
    Renders an ego network using Plotly.
    The central node is the selected company. Neighbors are arranged in a circle around it.
    """
    import math
    
    if not connections:
        return None
        
    fig = go.Figure()
    
    # Coordinates
    nodes_x = [0.0]
    nodes_y = [0.0]
    node_names = [company_name.title()]
    node_colors = ["#4F46E5"]  # Primary color for central node
    node_sizes = [40]
    node_texts = ["Target Company"]
    
    # Calculate positions for neighbors in a circle
    n_neighbors = len(connections)
    radius = 2.0
    
    for i, conn in enumerate(connections):
        angle = (i * 2 * math.pi) / n_neighbors
        nx_val = radius * math.cos(angle)
        ny_val = radius * math.sin(angle)
        
        neighbor = conn['neighbor'].title()
        conf = conn['confidence']
        
        nodes_x.append(nx_val)
        nodes_y.append(ny_val)
        node_names.append(neighbor)
        
        # Color intensity based on confidence
        node_colors.append(f"rgba(16, 185, 129, {max(0.3, conf)})")
        node_sizes.append(25)
        node_texts.append(f"Confidence: {conf:.2f}")
        
        # Add Edge
        fig.add_trace(go.Scatter(
            x=[0.0, nx_val],
            y=[0.0, ny_val],
            mode="lines",
            line=dict(width=conf * 5, color="rgba(156, 163, 175, 0.5)"),
            hoverinfo="none",
            showlegend=False
        ))
        
    # Add Nodes
    fig.add_trace(go.Scatter(
        x=nodes_x,
        y=nodes_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color="white")
        ),
        text=node_names,
        textposition="top center",
        hovertext=node_texts,
        hoverinfo="text",
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Node2Vec Ego Network for {company_name.title()}",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig


def plot_ego_network_pyvis(company_name, connections, explanations_dict=None):
    """
    Creates an interactive HTML graph using Pyvis.
    Returns the raw HTML string which can be rendered in Streamlit via components.html
    """
    from pyvis.network import Network
    import networkx as nx
    
    if not connections:
        return "<div>No connection data available to render graph.</div>"
        
    G = nx.Graph()
    G.add_node(company_name.title(), size=30, color="#4F46E5", title="Target Company")
    
    if explanations_dict is None:
        explanations_dict = {}
        
    for conn in connections:
        neighbor = conn['neighbor'].title()
        conf = conn['confidence']
        
        # Create rich tooltip from explanation if available
        tooltip = f"Confidence: {conf:.2f}"
        if neighbor in explanations_dict:
            exp = explanations_dict[neighbor]
            tooltip = f"Shared Articles: {exp['shared_articles_count']}\nPillar: {exp['dominant_pillar']}\nTone: {exp['avg_tone']:.2f}"
            
        G.add_node(neighbor, size=20, color="#10B981", title=tooltip)
        G.add_edge(company_name.title(), neighbor, value=conf * 10, title=tooltip)
        
    net = Network(height="500px", width="100%", bgcolor="white", font_color="black")
    net.from_nx(G)
    
    # Physics options for good layout
    net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {
          "size": 14,
          "face": "Outfit"
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "physics": {
        "repulsion": {
          "centralGravity": 0.2,
          "springLength": 150,
          "springConstant": 0.05,
          "nodeDistance": 150
        },
        "minVelocity": 0.75,
        "solver": "repulsion"
      }
    }
    """)
    
    html = net.generate_html()
    return html