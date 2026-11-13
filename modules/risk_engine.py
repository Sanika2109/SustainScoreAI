import pandas as pd
import numpy as np

def detect_risks(df_history, column="Overall_ESG", threshold_z=-2.0, periods=3):
    """
    Detects ESG risks:
    - Sudden Drop (Z-score based)
    - Continuous Decline (negative slope for periods)
    - High Volatility (Rolling Std Dev)
    """
    if df_history.empty or len(df_history) < 2:
        return []

    risks = []
    df = df_history.copy().sort_values("DATE")
    
    # 1. Sudden Drop
    mu = df[column].mean()
    sigma = df[column].std()
    
    if sigma > 0:
        last_val = df[column].iloc[-1]
        z_score = (last_val - mu) / sigma
        
        if z_score < threshold_z:
            risks.append({
                "type": "Sudden Drop",
                "severity": "Critical",
                "message": f"Sudden decline detected in {column} (Z={z_score:.2f}). Potential incident or negative media surge.",
                "pillar": column.split("_")[0]
            })

    # 2. Continuous Decline
    if len(df) >= periods:
        recent_vals = df[column].tail(periods)
        is_declining = all(recent_vals.iloc[i] > recent_vals.iloc[i+1] for i in range(len(recent_vals)-1))
        
        if is_declining:
            risks.append({
                "type": "Structural Decline",
                "severity": "Watch",
                "message": f"{column} has declined for {periods} consecutive periods. Consistent weakening of sustainability profile.",
                "pillar": column.split("_")[0]
            })

    # 3. High Volatility
    if len(df) >= 5:
        rolling_std = df[column].rolling(5).std().iloc[-1]
        baseline_std = df[column].std()
        
        if rolling_std > 1.5 * baseline_std:
            risks.append({
                "type": "High Volatility",
                "severity": "Warning",
                "message": f"Excessive volatility in {column} detected. Media sentiment is unstable.",
                "pillar": column.split("_")[0]
            })

    return risks
