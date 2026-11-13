"""
graph_explainability.py
=======================
Explainable AI (XAI) engine for Node2Vec graph embeddings.
Provides human-readable context on why two companies are clustered
by examining the underlying shared articles and ESG themes.
"""

import pandas as pd
import numpy as np

def get_top_connections(company, df_conn, top_n=5):
    """
    Extracts the top N closest neighbors for a specific company from the connections DataFrame.
    """
    if df_conn is None or df_conn.empty or "company" not in df_conn.columns:
        return []

    # Find the row for the specific company
    company_row = df_conn[df_conn["company"].str.lower() == company.lower()]
    if company_row.empty:
        return []

    company_row = company_row.iloc[0]
    
    connections = []
    for i in range(25):  # The model outputs up to 25 nearest neighbors
        rec_col = f"n{i}_rec"
        conf_col = f"n{i}_conf"
        
        if rec_col in company_row and conf_col in company_row:
            rec = company_row[rec_col]
            conf = company_row[conf_col]
            if pd.notna(rec) and pd.notna(conf):
                connections.append({
                    "neighbor": rec,
                    "confidence": float(conf)
                })
    
    # Sort by confidence descending, return top N
    connections = sorted(connections, key=lambda x: x["confidence"], reverse=True)
    return connections[:top_n]


def explain_connection(company_a, company_b, df_data, original_confidence=None, exclude_pillar=None, exclude_sentiment=None):
    """
    Extracts the shared articles between two companies and summarizes
    the ESG themes to explain why they are connected in the graph.
    Supports Counterfactual Analysis by excluding specific pillars or sentiments.
    """
    if df_data is None or df_data.empty or "URL" not in df_data.columns or "Organization" not in df_data.columns:
        return {
            "shared_articles_count": 0,
            "explanation": "No article data available to explain this connection.",
            "dominant_pillar": "N/A",
            "avg_tone": 0.0,
            "simulated_confidence": original_confidence or 0.0
        }

    # Filter data for both companies
    df_a = df_data[df_data["Organization"].str.lower() == company_a.lower()]
    df_b = df_data[df_data["Organization"].str.lower() == company_b.lower()]

    # Find shared URLs
    urls_a = set(df_a["URL"].dropna())
    urls_b = set(df_b["URL"].dropna())
    shared_urls = urls_a.intersection(urls_b)
    
    original_shared_count = len(shared_urls)
    if original_shared_count == 0:
        return {
            "shared_articles_count": 0,
            "explanation": f"**{company_a.title()}** and **{company_b.title()}** do not share direct article mentions in the current dataset.",
            "dominant_pillar": "N/A",
            "avg_tone": 0.0,
            "simulated_confidence": original_confidence or 0.0
        }

    # Extract the shared articles
    df_shared = df_data[df_data["URL"].isin(shared_urls)]
    tone_col = "AvgTone" if "AvgTone" in df_shared.columns else "Tone"
    
    # Apply Counterfactual Filters
    if exclude_pillar:
        if exclude_pillar == "Environmental" and "E" in df_shared.columns:
            df_shared = df_shared[df_shared["E"] != True]
        elif exclude_pillar == "Social" and "S" in df_shared.columns:
            df_shared = df_shared[df_shared["S"] != True]
        elif exclude_pillar == "Governance" and "G" in df_shared.columns:
            df_shared = df_shared[df_shared["G"] != True]
            
    if exclude_sentiment and tone_col in df_shared.columns:
        if exclude_sentiment == "Negative":
            df_shared = df_shared[df_shared[tone_col] >= -1.0]
        elif exclude_sentiment == "Positive":
            df_shared = df_shared[df_shared[tone_col] <= 1.0]
            
    shared_count = len(df_shared)
    
    # Calculate simulated confidence linearly based on remaining articles
    simulated_confidence = original_confidence or 0.0
    if original_shared_count > 0:
        simulated_confidence = simulated_confidence * (shared_count / original_shared_count)
        
    if shared_count == 0:
        return {
            "shared_articles_count": 0,
            "explanation": "After applying counterfactual filters, there are no shared articles remaining. The connection would likely be severed.",
            "dominant_pillar": "None",
            "avg_tone": 0.0,
            "sentiment_label": "None",
            "simulated_confidence": 0.0
        }

    # Calculate dominant pillar
    e_count = df_shared["E"].sum() if "E" in df_shared.columns else 0
    s_count = df_shared["S"].sum() if "S" in df_shared.columns else 0
    g_count = df_shared["G"].sum() if "G" in df_shared.columns else 0
    
    pillar_counts = {"Environmental": e_count, "Social": s_count, "Governance": g_count}
    dominant_pillar = max(pillar_counts, key=pillar_counts.get)
    
    if pillar_counts[dominant_pillar] == 0:
        dominant_pillar = "General ESG"

    # Calculate average tone
    if tone_col in df_shared.columns:
        avg_tone = df_shared[tone_col].mean()
    else:
        avg_tone = 0.0

    sentiment_label = "Negative" if avg_tone < -1.0 else "Positive" if avg_tone > 1.0 else "Neutral"
    
    cf_text = ""
    if exclude_pillar or exclude_sentiment:
        cf_text = " **(Simulated Counterfactual)**"
        
    explanation = (
        f"**{company_a.title()}** and **{company_b.title()}** share **{shared_count}** co-mentioned articles{cf_text}. "
        f"The overlapping coverage is primarily focused on the **{dominant_pillar}** pillar, "
        f"with a **{sentiment_label}** average tone ({avg_tone:+.2f})."
    )

    return {
        "shared_articles_count": shared_count,
        "explanation": explanation,
        "dominant_pillar": dominant_pillar,
        "avg_tone": avg_tone,
        "sentiment_label": sentiment_label,
        "simulated_confidence": simulated_confidence
    }

