import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

def predict_future_scores(df_history, periods=4, column="Overall_ESG"):
    """
    Predicts future scores using Linear Regression.
    Returns a dataframe with 'DATE', 'Predicted_Score', 'Lower_CI', 'Upper_CI'
    """
    if df_history.empty or len(df_history) < 3:
        return pd.DataFrame()

    df = df_history.copy().sort_values("DATE")
    df['Days'] = (df['DATE'] - df['DATE'].min()).dt.days
    
    X = df[['Days']].values
    y = df[column].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate future dates
    last_date = df['DATE'].max()
    freq = pd.infer_freq(df['DATE']) or 'D'
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
    
    future_days = (future_dates - df['DATE'].min()).days.values.reshape(-1, 1)
    predictions = model.predict(future_days)
    
    # Confidence Intervals (95%)
    # Standard error of the estimate
    y_pred = model.predict(X)
    residuals = y - y_pred
    n = len(y)
    p = 1 # number of predictors
    se = np.sqrt(np.sum(residuals**2) / (n - p - 1))
    
    # t-statistic for 95% CI
    t_stat = stats.t.ppf(0.975, n - p - 1)
    margin_error = t_stat * se
    
    df_pred = pd.DataFrame({
        'DATE': future_dates,
        'Predicted_Score': predictions,
        'Lower_CI': predictions - margin_error,
        'Upper_CI': predictions + margin_error,
        'is_forecast': True
    })
    
    return df_pred

def get_forecast_summary(df_history, df_forecast, column="Overall_ESG"):
    """
    Calculates percentage change from last historical value to last forecast.
    """
    if df_history.empty or df_forecast.empty:
        return 0, "stable"
    
    last_actual = df_history[column].iloc[-1]
    last_pred = df_forecast['Predicted_Score'].iloc[-1]
    
    pct_change = ((last_pred - last_actual) / (abs(last_actual) + 1e-8)) * 100
    sentiment = "improving" if pct_change > 2 else ("declining" if pct_change < -2 else "stable")
    
    return float(pct_change), sentiment
