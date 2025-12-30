import numpy as np
def forecast_with_uncertainty(device_payload, horizon=30):
    hist = device_payload.get("sensor_timeseries", [100.0]*5)
    # Simple linear regression projection
    x = np.arange(len(hist))
    y = np.array(hist)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    future_x = np.arange(len(hist), len(hist)+horizon)
    forecast = p(future_x).tolist()
    
    # Uncertainty
    sigma = np.std(hist) if len(hist) > 1 else 0.5
    upper = [v + (1.96*sigma) for v in forecast]
    lower = [v - (1.96*sigma) for v in forecast]
    
    return {"mean_forecast": forecast, "upper_bound": upper, "lower_bound": lower, "history_used": hist}
