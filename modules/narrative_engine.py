import pandas as pd
import numpy as np

def generate_performance_story(df_history, column="Overall_ESG"):
    """
    Identifies significant 'Turning Points' in the timeline and generates a story.
    Returns a list of dicts with 'DATE', 'Event', and 'Score'.
    """
    if df_history.empty or len(df_history) < 5:
        return []

    df = df_history.copy().sort_values("DATE")
    
    # 1. Detect Peak and Lowest
    peak_idx = df[column].idxmax()
    worst_idx = df[column].idxmin()
    
    # 2. Detect Momentum Shift
    df['Momentum'] = df[column].diff()
    df['Momentum_Shift'] = df['Momentum'].diff()
    
    # Significant increase / decrease
    increase_idx = df[column].diff().idxmax()
    decrease_idx = df[column].diff().idxmin()
    
    story_points = []
    
    # Add events
    story_points.append({
        "DATE": df.loc[peak_idx, "DATE"],
        "Event": "Peak Performance",
        "Description": f"ESG score reached historical peak of {df.loc[peak_idx, column]:.3f}. Significant increase in article sentiment and mentions.",
        "Score": df.loc[peak_idx, column]
    })
    
    story_points.append({
        "DATE": df.loc[worst_idx, "DATE"],
        "Event": "Critical Drop",
        "Description": f"ESG score fell to local trough of {df.loc[worst_idx, column]:.3f}. Suggests potential governance or environmental controversy.",
        "Score": df.loc[worst_idx, column]
    })
    
    if increase_idx != peak_idx:
        story_points.append({
            "DATE": df.loc[increase_idx, "DATE"],
            "Event": "Rapid Recovery",
            "Description": "ESG momentum turned strongly positive. Indication of improving media tone or structural sustainability gains.",
            "Score": df.loc[increase_idx, column]
        })
        
    if decrease_idx != worst_idx:
        story_points.append({
            "DATE": df.loc[decrease_idx, "DATE"],
            "Event": "Sudden Weakening",
            "Description": "ESG score dropped significantly. Monitoring of risk alerts is recommended.",
            "Score": df.loc[decrease_idx, column]
        })

    # Sort story points by date
    story_points.sort(key=lambda x: x['DATE'])
    
    return story_points

def format_story_as_text(story_points):
    """
    Formats the narrative into a user-friendly list.
    """
    if not story_points:
        return "No significant ESG performance events detected in this period."
    
    text = ""
    for point in story_points:
        text += f"• **{point['DATE'].strftime('%B %Y')}**: {point['Event']} - {point['Description']}\n"
    
    return text
