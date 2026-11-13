r"""
esg_engine.py
=============
Single Source of Truth for all ESG calculations.

FORMULAS (Industry Standard):
  1. Pillar Score (z-normalized signals):
     E_score = Z(NumArticles) * 0.6 + Z(AvgTone) * 0.4
     S_score = Z(NumMentions) * 0.5 + Z(AvgTone) * 0.5
     G_score = Z(AvgTone) * 0.7 + Z(NumArticles) * 0.3
  2. Weighted Composite ESG:
     Overall_ESG = \sum (P_i \cdot W_i) = E \cdot 0.4 + S \cdot 0.3 + G \cdot 0.3
  3. Display Scale (0-100):
     Display = \min(\max(50 + Overall\_ESG \cdot 15, 0), 100)
"""

import pandas as pd
import numpy as np
from datetime import datetime, date

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════

PILLAR_WEIGHTS = {"E_score": 0.4, "S_score": 0.3, "G_score": 0.3}

PILLAR_LABELS = {
    "E_score": "Environmental",
    "S_score": "Social",
    "G_score": "Governance",
    "Overall_ESG": "Overall ESG",
}

PILLAR_COLORS = {
    "E_score": "#10B981",   # Emerald Green
    "S_score": "#3B82F6",   # Ocean Blue
    "G_score": "#F97316",   # Warm Orange
    "Overall_ESG": "#6366F1",  # Indigo / Purple
}

FORMULAS = {
    "E_score": "E = Norm(NumArticles) × 0.6 + Norm(AvgTone) × 0.4",
    "S_score": "S = Norm(NumMentions) × 0.5 + Norm(AvgTone) × 0.5",
    "G_score": "G = Norm(AvgTone) × 0.7 + Norm(NumArticles) × 0.3",
    "Overall_ESG": "Overall = E × 0.4 + S × 0.3 + G × 0.3",
}


# ═══════════════════════════════════════════════════════════════════
#  NORMALIZATION
# ═══════════════════════════════════════════════════════════════════

def normalize(series: pd.Series) -> pd.Series:
    """
    Standard Z-score normalization: z = (x - μ) / σ.
    Handles zero-variance cases by returning zero-centered series.
    """
    if series.empty or len(series) < 2:
        return series - series.mean() if not series.empty else series
    
    mu = series.mean()
    sigma = series.std()
    
    if sigma < 1e-12 or np.isnan(sigma):
        return series - mu
        
    return (series - mu) / sigma


# ═══════════════════════════════════════════════════════════════════
#  SCALING
# ═══════════════════════════════════════════════════════════════════

def to_display_score(mean_raw: float) -> float:
    """Scales a Z-normalized mean to a 0-100 display range (Neutral baseline = 50.0)."""
    score = float(np.clip(50 + mean_raw * 15, 0, 100))
    # Handle negative zero
    if abs(score) < 1e-6:
        return 0.0
    return score


# ═══════════════════════════════════════════════════════════════════
#  DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps fallback columns so that NumArticles, NumMentions, AvgTone
    always exist. Converts to numeric.
    """
    df = df.copy()

    # Column mapping (fallbacks)
    if "NumArticles" not in df.columns:
        df["NumArticles"] = df.get("WordCount", pd.Series(1, index=df.index))
    if "NumMentions" not in df.columns:
        df["NumMentions"] = df.get("ActivityDensity", pd.Series(1, index=df.index))
    if "AvgTone" not in df.columns:
        df["AvgTone"] = df.get("Tone", pd.Series(0, index=df.index))

    # Force numeric
    for col in ["NumArticles", "NumMentions", "AvgTone", "Tone"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where tone is completely missing
    tone_col = "AvgTone" if "AvgTone" in df.columns else "Tone"
    if tone_col in df.columns:
        df = df.dropna(subset=[tone_col])

    # Ensure DATE is datetime
    if "DATE" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    
    return df


def preprocess_company_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Ensure columns exist
    2. Handle NaN / duplicates
    3. Compute pillar scores
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = ensure_columns(df)
    df = df.drop_duplicates(subset=["DATE", "URL"], keep="first") if "URL" in df.columns else df
    df = compute_pillar_scores(df)
    return df


# ═══════════════════════════════════════════════════════════════════
#  PILLAR SCORE CALCULATION
# ═══════════════════════════════════════════════════════════════════

def compute_pillar_scores(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    Computes E, S, G scores and Overall Composite.

    Formula:
    $$E = 0.6 \cdot Z(Articles) + 0.4 \cdot Z(Tone)$$
    $$S = 0.5 \cdot Z(Mentions) + 0.4 \cdot Z(Tone)$$
    $$G = 0.7 \cdot Z(Tone) + 0.3 \cdot Z(Articles)$$
    $$Overall = 0.4E + 0.3S + 0.3G$$
    """
    if df.empty:
        return df

    df = df.copy()

    norm_articles = normalize(df["NumArticles"]) if "NumArticles" in df.columns else 0.0
    norm_mentions = normalize(df["NumMentions"]) if "NumMentions" in df.columns else 0.0
    norm_tone = normalize(df["AvgTone"]) if "AvgTone" in df.columns else 0.0

    df["E_score"] = (norm_articles * 0.6) + (norm_tone * 0.4)
    df["S_score"] = (norm_mentions * 0.5) + (norm_tone * 0.5)
    df["G_score"] = (norm_tone * 0.7) + (norm_articles * 0.3)
    df["Overall_ESG"] = (
        df["E_score"] * PILLAR_WEIGHTS["E_score"]
        + df["S_score"] * PILLAR_WEIGHTS["S_score"]
        + df["G_score"] * PILLAR_WEIGHTS["G_score"]
    )
    return df


def compute_global_company_scores(df_all: pd.DataFrame, company: str) -> pd.DataFrame:
    """
    Computes ESG scores for a specific company, but normalizes
    using the GLOBAL population (all companies). This ensures
    individual company means differ meaningfully.
    """
    df_all = ensure_columns(df_all)

    # Normalize across ALL companies
    norm_articles = normalize(df_all["NumArticles"]) if "NumArticles" in df_all.columns else 0.0
    norm_mentions = normalize(df_all["NumMentions"]) if "NumMentions" in df_all.columns else 0.0
    norm_tone = normalize(df_all["AvgTone"]) if "AvgTone" in df_all.columns else 0.0

    df_all = df_all.copy()
    df_all["E_score"] = (norm_articles * 0.6) + (norm_tone * 0.4)
    df_all["S_score"] = (norm_mentions * 0.5) + (norm_tone * 0.5)
    df_all["G_score"] = (norm_tone * 0.7) + (norm_articles * 0.3)
    df_all["Overall_ESG"] = (
        df_all["E_score"] * PILLAR_WEIGHTS["E_score"]
        + df_all["S_score"] * PILLAR_WEIGHTS["S_score"]
        + df_all["G_score"] * PILLAR_WEIGHTS["G_score"]
    )

    # Filter to the specific company
    df_company = df_all[df_all["Organization"].str.lower() == company.lower()].copy()
    return df_company


# ═══════════════════════════════════════════════════════════════════
#  COMPANY SUMMARY
# ═══════════════════════════════════════════════════════════════════

def compute_company_summary(df_company: pd.DataFrame, df_global_scored: pd.DataFrame = None) -> dict:
    """
    Returns a summary dict for a single company with mean scores.
    
    If df_global_scored is provided (company data scored with global normalization),
    the means reflect the company's position relative to the entire universe.
    Otherwise falls back to local scores.
    """
    source = df_global_scored if df_global_scored is not None and not df_global_scored.empty else df_company

    if source.empty:
        return {
            "e_mean": 0, "s_mean": 0, "g_mean": 0, "overall_mean": 0,
            "display_esg": 50, "trend": "stable", "n_articles": 0,
        }

    e_mean = source["E_score"].mean()
    s_mean = source["S_score"].mean()
    g_mean = source["G_score"].mean()
    overall_mean = source["Overall_ESG"].mean()

    # Scale to user-friendly 0-100 range
    display_esg = to_display_score(overall_mean)

    # Determine trend from local per-company data (captures within-company dynamics)
    trend_source = df_company if not df_company.empty else source
    if len(trend_source) >= 4 and "Overall_ESG" in trend_source.columns:
        half = len(trend_source) // 2
        sorted_df = trend_source.sort_values("DATE")
        first_half = sorted_df["Overall_ESG"].iloc[:half].mean()
        second_half = sorted_df["Overall_ESG"].iloc[half:].mean()
        diff = second_half - first_half
        if diff > 0.05:
            trend = "improving"
        elif diff < -0.05:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "stable"

    return {
        "e_mean": float(e_mean),
        "s_mean": float(s_mean),
        "g_mean": float(g_mean),
        "overall_mean": float(overall_mean),
        "display_esg": display_esg,
        "trend": trend,
        "n_articles": len(df_company) if not df_company.empty else len(source),
    }


# ═══════════════════════════════════════════════════════════════════
#  INDUSTRY BENCHMARK
# ═══════════════════════════════════════════════════════════════════

def compute_industry_benchmark(df_all: pd.DataFrame) -> dict:
    """
    Computes industry-average scores from ALL companies' data.
    """
    df_all = ensure_columns(df_all)
    df_all = compute_pillar_scores(df_all)

    return {
        "e_mean": float(df_all["E_score"].mean()),
        "s_mean": float(df_all["S_score"].mean()),
        "g_mean": float(df_all["G_score"].mean()),
        "overall_mean": float(df_all["Overall_ESG"].mean()),
        "e_display": to_display_score(df_all["E_score"].mean()),
        "s_display": to_display_score(df_all["S_score"].mean()),
        "g_display": to_display_score(df_all["G_score"].mean()),
        "overall_display": to_display_score(df_all["Overall_ESG"].mean()),
    }


# ═══════════════════════════════════════════════════════════════════
#  COMPANY RANKING
# ═══════════════════════════════════════════════════════════════════

def rank_companies(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks all companies by their mean Overall_ESG score.
    Returns a DataFrame with columns:
        Organization, E_mean, S_mean, G_mean, Overall_mean, Rank
    """
    df_all = ensure_columns(df_all)
    df_all = compute_pillar_scores(df_all)

    ranked = (
        df_all.groupby("Organization")
        .agg(
            E_mean=("E_score", "mean"),
            S_mean=("S_score", "mean"),
            G_mean=("G_score", "mean"),
            Overall_mean=("Overall_ESG", "mean"),
            ArticleCount=("E_score", "count"),
        )
        .reset_index()
        .sort_values("Overall_mean", ascending=False)
        .reset_index(drop=True)
    )
    ranked["Rank"] = ranked.index + 1
    ranked["Display_ESG"] = np.clip(50 + ranked["Overall_mean"] * 15, 0, 100)
    return ranked


# ═══════════════════════════════════════════════════════════════════
#  MOMENTUM & VOLATILITY
# ═══════════════════════════════════════════════════════════════════

def compute_momentum(series: pd.Series, window_pct: float = 0.2) -> float:
    """
    Momentum = μ_recent - μ_baseline.
    Uses the first/last `window_pct` of sorted data.
    """
    n = len(series)
    if n < 2:
        return 0.0
    window = max(1, int(n * window_pct))

    sorted_vals = series.sort_index()
    baseline = sorted_vals.head(window).mean()
    recent = sorted_vals.tail(window).mean()
    return float(recent - baseline)


def compute_volatility(series: pd.Series) -> float:
    """Standard deviation of normalized values."""
    if series.empty or len(series) < 2:
        return 0.0
    return float(normalize(series).std())


# ═══════════════════════════════════════════════════════════════════
#  OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_outliers(df_ranked: pd.DataFrame, column: str = "Overall_mean",
                    method: str = "iqr") -> pd.DataFrame:
    """
    Detects outlier companies using IQR method.
    Returns the input DataFrame with an 'is_outlier' boolean column.
    """
    if df_ranked.empty or column not in df_ranked.columns:
        df_ranked["is_outlier"] = False
        return df_ranked

    df = df_ranked.copy()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df["is_outlier"] = (df[column] < lower) | (df[column] > upper)
    df["outlier_type"] = np.where(
        df[column] < lower, "Low Outlier",
        np.where(df[column] > upper, "High Outlier", "Normal")
    )
    return df


# ═══════════════════════════════════════════════════════════════════
#  EXECUTIVE METRICS (replaces compute_executive_metrics in app.py)
# ═══════════════════════════════════════════════════════════════════

def compute_executive_metrics(df_company: pd.DataFrame, df_all: pd.DataFrame) -> dict:
    """
    Computes the four executive KPI values for the dashboard header.
    Returns dict with scaled display scores for both company and industry.
    """
    company_summary = compute_company_summary(df_company)
    # Get global scores for company to ensure diffs
    df_global_scored = compute_global_company_scores(df_all, df_company["Organization"].iloc[0] if not df_company.empty else "")
    company_summary_global = compute_company_summary(df_company, df_global_scored)
    
    industry = compute_industry_benchmark(df_all)

    # Momentum from tone
    tone_col = "AvgTone" if "AvgTone" in df_company.columns else "Tone"
    if tone_col in df_company.columns:
        momentum = compute_momentum(df_company[tone_col])
        volatility = compute_volatility(df_company[tone_col])
    else:
        momentum = 0.0
        volatility = 0.0

    # Scaled Scores (0-100)
    # The mean of Z-normalized scores is 0, so Industry displayed score will be 50.0
    ind_display = float(np.clip(50 + industry["overall_mean"] * 15, 0, 100))
    comp_display = company_summary_global["display_esg"]

    # Simple gap in display units
    gap = comp_display - ind_display
    
    # Handle negative zero for display
    if abs(ind_display) < 1e-6: ind_display = 0.0
    if abs(momentum) < 1e-6: momentum = 0.0

    return {
        "composite_esg": comp_display,
        "industry_esg": ind_display,
        "momentum": momentum,
        "volatility": volatility,
        "gap": gap,
        "trend": company_summary_global["trend"],
        "n_articles": company_summary_global["n_articles"],
    }

# ═══════════════════════════════════════════════════════════════════
#  TIME-BASED ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def compute_esg_aggregates(df: pd.DataFrame, freq: str = 'weekly') -> pd.DataFrame:
    """
    Standardized aggregation function for Weekly and Monthly Data.
    Guarantees exact matching on date definitions rather than implicit pandas resampling.
    """
    if df.empty or "DATE" not in df.columns:
        return pd.DataFrame()
        
    df = df.copy()
    # ensure dataframe is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"])

    # Ensure scores are computed if missing
    score_cols = ["E_score", "S_score", "G_score", "Overall_ESG"]
    if not all(col in df.columns for col in score_cols):
        df = compute_pillar_scores(df)

    if freq == 'weekly':
        df['year'] = df['DATE'].dt.isocalendar().year
        df['week'] = df['DATE'].dt.isocalendar().week
        
        # Mean for scores
        aggregated_df = df.groupby(['year', 'week'], as_index=False)[score_cols].mean()
        
        # Statistical Variance (Spread/Risk) for each pillar
        var_df = df.groupby(['year', 'week'], as_index=False)[score_cols].var().fillna(0)
        for col in score_cols:
            aggregated_df[f"{col}_Stat_Var"] = var_df[col]
            
        # Add a standard 'DATE' (Monday of the week)
        aggregated_df['DATE'] = aggregated_df.apply(
            lambda r: pd.Timestamp(date.fromisocalendar(int(r['year']), int(r['week']), 1)), axis=1
        )
    elif freq == 'monthly':
        df['year'] = df['DATE'].dt.year
        df['month'] = df['DATE'].dt.month
        
        # Mean for scores
        aggregated_df = df.groupby(['year', 'month'], as_index=False)[score_cols].mean()
        
        # Statistical Variance (Spread/Risk) for each pillar
        var_df = df.groupby(['year', 'month'], as_index=False)[score_cols].var().fillna(0)
        for col in score_cols:
            aggregated_df[f"{col}_Stat_Var"] = var_df[col]
            
        # Add a standard 'DATE' (1st day of the month)
        aggregated_df['DATE'] = pd.to_datetime(aggregated_df.assign(day=1)[['year', 'month', 'day']])
    else:
        return pd.DataFrame()
        
    return aggregated_df.sort_values(by="DATE").reset_index(drop=True)


def compute_growth_metrics(df_resampled: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates period-over-period growth percentages and delta.

    Design decisions
    ----------------
    * The first row has no previous period, so Performance_Delta and
      Growth_Pct are left as NaN. The render layer detects NaN and
      shows "No prior period" instead of a misleading value.
    * ESG pillars are Z-normalised and float around 0, so dividing by
      a near-zero previous value can produce unbounded percentages
      (e.g. +426 %).  A symmetric mid-point denominator is used:

          Growth_Pct = delta / mean(|prev|, |current|) * 100

      The result is additionally clipped to [-100, +100] % so that
      legitimate large swings are shown as ±100 % rather than noise.
    """
    if df_resampled.empty:
        return df_resampled

    df = df_resampled.copy()
    score_cols = ["E_score", "S_score", "G_score", "Overall_ESG"]

    for col in score_cols:
        if col not in df.columns:
            continue

        prev = df[col].shift(1)
        delta = df[col] - prev

        # Performance Delta — NaN for first row (no previous period)
        df[f"{col}_Performance_Delta"] = delta

        df[f"{col}_Status"] = np.where(
            prev.isna(),
            "No prior period",
            "OK"
        )

        # Growth % — symmetric denominator, clipped, NaN preserved for first row
        midpoint = (prev.abs() + df[col].abs()) / 2.0 + 1e-8
        raw_pct = (delta / midpoint) * 100.0
        df[f"{col}_Growth_Pct"] = raw_pct.clip(-100.0, 100.0)

    return df


def detect_esg_anomalies(df_resampled: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """
    Detects sudden drops or spikes in ESG scores using Z-score of the time series.
    threshold: Number of standard deviations to trigger anomaly.
    """
    if df_resampled.empty or len(df_resampled) < 3:
        df_resampled["is_anomaly"] = False
        return df_resampled

    df = df_resampled.copy()
    score = df["Overall_ESG"]
    
    # Simple Z-score of the scores themselves over time
    mu = score.mean()
    sigma = score.std()
    
    if sigma == 0: sigma = 1e-8
    
    df["z_score"] = (score - mu) / sigma
    df["is_anomaly"] = df["z_score"].abs() > threshold
    df["anomaly_type"] = np.where(df["z_score"] > threshold, "Spike", 
                         np.where(df["z_score"] < -threshold, "Drop", "Normal"))
    
    return df


def get_period_performance_summary(df_resampled: pd.DataFrame) -> dict:
    """
    Finds best and worst performing periods.
    """
    if df_resampled.empty:
        return {"best_period": None, "worst_period": None}

    best_idx = df_resampled["Overall_ESG"].idxmax()
    worst_idx = df_resampled["Overall_ESG"].idxmin()
    
    return {
        "best_period": df_resampled.loc[best_idx, "DATE"],
        "best_score": df_resampled.loc[best_idx, "Overall_ESG"],
        "worst_period": df_resampled.loc[worst_idx, "DATE"],
        "worst_score": df_resampled.loc[worst_idx, "Overall_ESG"],
    }