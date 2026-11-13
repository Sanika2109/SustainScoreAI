"""
reprocess_gdelt.py
==================
Reprocesses a GDELT dataset to match the dec30_to_jan12 pivoted schema
expected by download_data.py and app.py.

Usage:
    python reprocess_gdelt.py                           # defaults to Data/GDELT_20250101_20260329
    python reprocess_gdelt.py --dataset Data/GDELT_Live
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
import itertools
from datetime import datetime


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. LOAD COMPANY PROFILES (source of truth for tracked companies)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_tracked_companies(profile_path="company_profiles.json"):
    """Returns a set of lowercase company keys from company_profiles.json."""
    with open(profile_path, "r") as f:
        profiles = json.load(f)
    return set(k.strip().lower() for k in profiles.keys()) # already lowercase


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. FILTER & NORMALIZE RAW DATA
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def filter_and_normalize(data_path, tracked):
    """
    Reads data_as_csv.csv, filters to tracked companies, normalizes org names
    to lowercase, and standardizes the DATE column.
    """
    print(f"  Reading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    # Normalize Organization to lowercase for matching
    df["Organization"] = df["Organization"].astype(str).str.strip().str.lower()

    # Filter to tracked companies only
    df = df[df["Organization"].isin(tracked)].copy()
    print(f"  Filtered rows (tracked orgs only): {len(df):,}")

    if df.empty:
        print("  WARNING: No rows matched tracked companies!")
        return df

    # ✅ FIX 1: Robust DATE parsing
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])
    df["DATE"] = df["DATE"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Ensure E/S/G are boolean-like (True/False or 1/0)
    for col in ["E", "S", "G"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # ✅ FIX 2: Ensure numeric consistency (VERY IMPORTANT)
    numeric_cols = [
        "Tone", "ESG_Score", "PositiveTone", "NegativeTone",
        "Polarity", "ActivityDensity", "SelfDensity"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid tone rows (critical for aggregates)
    df = df.dropna(subset=["Tone"])

    # ✅ FIX 3: Add missing metrics for standard consistency
    if "AvgTone" not in df.columns:
        df["AvgTone"] = df["Tone"]
    if "NumArticles" not in df.columns:
        df["NumArticles"] = 1
    if "NumMentions" not in df.columns:
        df["NumMentions"] = 1

    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. GENERATE ESG AGGREGATES  (pivoted format matching dec30_to_jan12)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_pivoted_aggregates(df):
    """
    Generates the 5 ESG aggregate CSVs in the PIVOTED format expected by the
    dashboard.
    
    Returns a dict of {filename: DataFrame}.
    """
    if df.empty:
        return {}

    # Parse date for grouping
    df["dt_obj"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["dt_obj"])
    df["date"] = df["dt_obj"].dt.date

    tone_col = "Tone"

    # â”€â”€ Industry Tone (benchmark = avg tone per day across ALL rows) â”€â”€
    industry_tone = (
        df.groupby("date")[tone_col]
        .mean()
        .reset_index()
        .rename(columns={tone_col: "industry_tone"})
    )

    # â”€â”€ Helper: build a pivoted daily score table â”€â”€
    def _pivoted_daily(subset, value_col=tone_col):
        """
        Pivots: rows=dates, columns=organizations, values=mean tone.
        Adds industry_tone column. Renames org cols to {org}_diff.
        """
        if subset.empty:
            return pd.DataFrame(columns=["date", "industry_tone"])

        daily = (
            subset.groupby(["date", "Organization"])[value_col]
            .mean()
            .reset_index()
        )
        pivoted = daily.pivot(index="date", columns="Organization", values=value_col)

        # Rename columns: "twitter" -> "twitter_diff"
        pivoted.columns = [
            f"{c.replace(' ', '_')}_diff" for c in pivoted.columns
        ]
        pivoted = pivoted.reset_index()

        # Merge industry tone
        pivoted = pivoted.merge(industry_tone, on="date", how="left")

        # Reorder so industry_tone comes right after date
        cols = ["date", "industry_tone"] + [
            c for c in pivoted.columns if c not in ("date", "industry_tone")
        ]
        return pivoted[cols]

    # â”€â”€ 1. Average ESG Scores (pivoted: companies as columns, E/S/G as rows) â”€â”€
    org_scores = df.groupby("Organization").agg(
        E_mean=(tone_col, lambda x: x[df.loc[x.index, "E"]].mean() if df.loc[x.index, "E"].any() else np.nan),
        S_mean=(tone_col, lambda x: x[df.loc[x.index, "S"]].mean() if df.loc[x.index, "S"].any() else np.nan),
        G_mean=(tone_col, lambda x: x[df.loc[x.index, "G"]].mean() if df.loc[x.index, "G"].any() else np.nan),
    ).T

    # Rename the index to match dec30 format
    org_scores.index = ["E", "S", "G"]
    org_scores.index.name = None  # Will become "Unnamed: 0" in CSV

    # Column names are organization names (already lowercase)
    avg_esg = org_scores.reset_index().rename(columns={"index": ""})

    # â”€â”€ 2. Overall Daily ESG Scores â”€â”€
    # Use ESG_Score if present, else use Tone
    score_col = "ESG_Score" if "ESG_Score" in df.columns else tone_col
    overall_daily = _pivoted_daily(df, value_col=score_col)

    # â”€â”€ 3. Daily E Score â”€â”€
    daily_e = _pivoted_daily(df[df["E"] == True])

    # â”€â”€ 4. Daily S Score â”€â”€
    daily_s = _pivoted_daily(df[df["S"] == True])

    # â”€â”€ 5. Daily G Score â”€â”€
    daily_g = _pivoted_daily(df[df["G"] == True])

    return {
        "average_esg_scores.csv": avg_esg,
        "overall_daily_esg_scores.csv": overall_daily,
        "daily_E_score.csv": daily_e,
        "daily_S_score.csv": daily_s,
        "daily_G_score.csv": daily_g,
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. GENERATE CONNECTIONS  (co-occurrence graph with 25 neighbors)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def generate_connections(df, n_neighbors=25):
    """
    Builds a co-occurrence graph from articles (same URL = co-mention),
    then extracts the top-N neighbors for each company.
    
    Output matches dec30_to_jan12 format:
        company, n0_rec, n0_conf, n1_rec, n1_conf, ...
    """
    print("  Computing co-occurrence connections...")

    # Find articles that mention multiple orgs
    url_orgs = df.groupby("URL")["Organization"].apply(list).reset_index()
    url_orgs = url_orgs[url_orgs["Organization"].apply(len) > 1]

    # Build edge weights
    edge_weights = {}
    for _, row in url_orgs.iterrows():
        orgs = sorted(set(str(o).strip().lower() for o in row["Organization"] if pd.notna(o))) # deduplicate within article
        for a, b in itertools.combinations(sorted(orgs), 2):
            key = (a, b)
            edge_weights[key] = edge_weights.get(key, 0) + 1

    # Build neighbor map: org -> [(neighbor, weight), ...] sorted by weight desc
    neighbor_map = {}
    for (a, b), w in edge_weights.items():
        neighbor_map.setdefault(a, []).append((b, w))
        neighbor_map.setdefault(b, []).append((a, w))
    # Normalize weights (0–1 scale)
    if edge_weights:
        max_w = max(edge_weights.values())
        edge_weights = {k: v / max_w for k, v in edge_weights.items()}

    for org in neighbor_map:
        neighbor_map[org].sort(key=lambda x: -x[1])

    # Build the connections DataFrame
    all_orgs = sorted(df["Organization"].unique())
    rows = []
    for org in all_orgs:
        row = {"company": org}
        neighbors = neighbor_map.get(org, [])
        for i in range(n_neighbors):
            if i < len(neighbors):
                row[f"n{i}_rec"] = neighbors[i][0]
                row[f"n{i}_conf"] = neighbors[i][1]
            else:
                row[f"n{i}_rec"] = ""
                row[f"n{i}_conf"] = 0
        rows.append(row)
    
    if not rows:
        return pd.DataFrame(columns=["company"])


    return pd.DataFrame(rows)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. GENERATE PCA EMBEDDINGS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_embeddings(df, n_components=10):
    """
    Creates a simple feature matrix from ESG signals and tone,
    then applies PCA to generate embedding vectors per organization.
    """
    print("  Computing PCA embeddings...")

    feature_cols = ["Tone", "PositiveTone", "NegativeTone", "Polarity",
                    "ActivityDensity", "SelfDensity"]
    safe_cols = [c for c in feature_cols if c in df.columns]

    if len(safe_cols) < 2:
        # Fallback: zero embeddings
        unique_orgs = df["Organization"].unique()
        embed_df = pd.DataFrame({"Organization": unique_orgs})
        for i in range(n_components):
            embed_df[f"c{i}"] = 0.0
        return embed_df

    # Compute per-org feature vectors
    org_features = df.groupby("Organization")[safe_cols].mean().fillna(0)

    # Add E/S/G ratios
    for col in ["E", "S", "G"]:
        if col in df.columns:
            ratio = df.groupby("Organization")[col].mean()
            org_features[f"{col}_ratio"] = ratio

    # Standardize
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(org_features.values)

    # PCA
    n_comp = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_comp)
    components = pca.fit_transform(X)

    embed_df = pd.DataFrame({"Organization": org_features.index})
    for i in range(n_comp):
        embed_df[f"c{i}"] = components[:, i]
    # Pad remaining columns with zeros if n_comp < n_components
    for i in range(n_comp, n_components):
        embed_df[f"c{i}"] = 0.0

    return embed_df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. MAIN PIPELINE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def reprocess(dataset_path):
    """Full reprocessing pipeline."""
    print(f"\n{'='*60}")
    print(f"  REPROCESSING: {dataset_path}")
    print(f"{'='*60}\n")

    data_csv = os.path.join(dataset_path, "data_as_csv.csv")
    esg_dir = os.path.join(dataset_path, "ESG")

    if not os.path.exists(data_csv):
        print(f"ERROR: {data_csv} not found!")
        return False

    os.makedirs(esg_dir, exist_ok=True)

    # Step 1: Load tracked companies
    tracked = load_tracked_companies()
    print(f"  Tracked companies: {len(tracked)}")

    # Step 2: Filter & normalize
    df = filter_and_normalize(data_csv, tracked)
    if df.empty:
        print("  ERROR: No data after filtering. Aborting.")
        return False

    # Step 3: Save filtered data_as_csv.csv
    print(f"  Saving filtered data ({len(df):,} rows)...")
    df.to_csv(data_csv, index=False)
    print(f"  âœ“ Saved {data_csv}")

    # Step 4: Generate pivoted ESG aggregates
    print("\n  Generating ESG aggregates (pivoted format)...")
    aggregates = generate_pivoted_aggregates(df)
    for filename, agg_df in aggregates.items():
        path = os.path.join(esg_dir, filename)
        agg_df.to_csv(path, index=False)
        print(f"  âœ“ {filename}: {agg_df.shape}")

    # Step 5: Generate connections
    conn_df = generate_connections(df)
    conn_path = os.path.join(dataset_path, "connections.csv")
    conn_df.to_csv(conn_path, index=False)
    print(f"  âœ“ connections.csv: {conn_df.shape}")

    # Step 6: Generate PCA embeddings
    try:
        embed_df = generate_embeddings(df)
    except ImportError:
        print("  WARNING: sklearn not available, generating zero embeddings")
        unique_orgs = df["Organization"].unique()
        embed_df = pd.DataFrame({"Organization": unique_orgs})
        for i in range(10):
            embed_df[f"c{i}"] = 0.0
    embed_path = os.path.join(dataset_path, "pca_embeddings.csv")
    embed_df.to_csv(embed_path, index=False)
    print(f"  âœ“ pca_embeddings.csv: {embed_df.shape}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  COMPLETE!")
    print(f"  Organizations: {df['Organization'].nunique()}")
    print(f"  Date range: {df['DATE'].min()} â†’ {df['DATE'].max()}")
    print(f"  Total rows: {len(df):,}")
    print(f"{'='*60}\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reprocess GDELT data to match dec30_to_jan12 schema")
    parser.add_argument("--dataset", type=str, default="Data/GDELT_20250101_20260329",
                        help="Path to the GDELT dataset directory")
    args = parser.parse_args()
    reprocess(args.dataset)

