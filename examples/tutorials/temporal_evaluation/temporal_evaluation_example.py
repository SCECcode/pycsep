"""
Temporal Forecast Evaluation - Multiple Models Comparison (GeoNet)
===================================================================

This example demonstrates how to compare multiple temporal probability forecasts
using ROC and Molchan diagrams with the GeoNet (NZ) earthquake catalog.
"""

import csep
from csep.utils import plots
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

# Dictionary of forecasts to compare: {name: filename}
# Add your forecast files here
FORECASTS = {
    'Model_A': 'temporal_forecast_500days.csv',
    'Model_B': 'temporal_forecast_model_b.csv',
    # 'EEPAS': 'eepas_forecast.csv',
}

# Time settings
START_TIME = '2016-01-01'
TIME_DELTA = '1D'

# Magnitude threshold
MIN_MAGNITUDE = 4.0

# Region settings
USE_BOUNDING_BOX = True
MIN_LATITUDE = -47.0
MAX_LATITUDE = -34.0
MIN_LONGITUDE = 164.0
MAX_LONGITUDE = 180.0

# Polygon (if USE_BOUNDING_BOX = False)
POLYGON = [
    [174.0, -42.0],
    [176.0, -42.0],
    [176.0, -40.0],
    [174.0, -40.0],
    [174.0, -42.0],
]

# Plot styling
COLORS = ['black', 'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']

# =============================================================================
# Load forecasts
# =============================================================================

print("="*70)
print("TEMPORAL FORECAST COMPARISON - NEW ZEALAND (GEONET)")
print("="*70)

forecasts_data = {}
for name, filename in FORECASTS.items():
    path = os.path.join(script_dir, filename)
    if os.path.exists(path):
        forecasts_data[name] = csep.load_temporal_forecast(path)
        print(f"  Loaded: {name} ({forecasts_data[name]['metadata']['n_time_windows']} days)")
    else:
        print(f"  WARNING: {filename} not found, skipping {name}")

if not forecasts_data:
    print("ERROR: No forecasts loaded!")
    exit(1)

# Get number of time windows from first forecast
n_days = list(forecasts_data.values())[0]['metadata']['n_time_windows']

# Calculate end date
start_dt = pd.to_datetime(START_TIME)
end_dt = start_dt + pd.Timedelta(TIME_DELTA) * n_days
print(f"\nForecast period: {START_TIME} to {end_dt.strftime('%Y-%m-%d')}")

# =============================================================================
# Query GeoNet catalog
# =============================================================================

print(f"\nQuerying GeoNet catalog...")
print(f"  Magnitude >= {MIN_MAGNITUDE}")

if USE_BOUNDING_BOX:
    print(f"  Region: Lat [{MIN_LATITUDE}, {MAX_LATITUDE}], Lon [{MIN_LONGITUDE}, {MAX_LONGITUDE}]")
    catalog = csep.query_gns(
        start_time=datetime.fromisoformat(START_TIME),
        end_time=end_dt.to_pydatetime(),
        min_magnitude=MIN_MAGNITUDE,
        min_latitude=MIN_LATITUDE,
        max_latitude=MAX_LATITUDE,
        min_longitude=MIN_LONGITUDE,
        max_longitude=MAX_LONGITUDE,
        verbose=False
    )
else:
    print(f"  Region: Custom polygon")
    catalog = csep.query_gns(
        start_time=datetime.fromisoformat(START_TIME),
        end_time=end_dt.to_pydatetime(),
        min_magnitude=MIN_MAGNITUDE,
        min_latitude=-50.0, max_latitude=-30.0,
        min_longitude=160.0, max_longitude=180.0,
        verbose=False
    )
    from shapely.geometry import Point, Polygon
    poly = Polygon(POLYGON)
    lons, lats = catalog.get_longitudes(), catalog.get_latitudes()
    inside_mask = np.array([poly.contains(Point(lon, lat)) for lon, lat in zip(lons, lats)])
    catalog = catalog.filter(inside_mask)

print(f"  Found {catalog.event_count} M{MIN_MAGNITUDE}+ events")

# =============================================================================
# Compute observations (shared by all forecasts)
# =============================================================================

times = list(forecasts_data.values())[0]['times']
observations = csep.compute_temporal_observations(
    catalog, times,
    start_time=START_TIME,
    time_delta=TIME_DELTA,
    magnitude_min=MIN_MAGNITUDE
)

n_events = np.sum(observations)
print(f"  Events in {n_days} days: {n_events}")

# =============================================================================
# Evaluate all forecasts and plot
# =============================================================================

print(f"\nEvaluating {len(forecasts_data)} forecast(s)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
results = []

for i, (name, data) in enumerate(forecasts_data.items()):
    color = COLORS[i % len(COLORS)]
    linestyle = LINESTYLES[i % len(LINESTYLES)]
    
    probs = data['probabilities']
    
    # ROC
    ax1, auc = plots.plot_temporal_ROC_diagram(
        probs, observations,
        name=name,
        axes=axes[0],
        show=False,
        plot_uniform=(i == 0),  # Only plot uniform line once
        plot_args={
            'linecolor': color,
            'linestyle': linestyle,
            'title': f'Temporal ROC Diagram (M{MIN_MAGNITUDE}+)'
        }
    )
    
    # Molchan
    ax2, ass, sigma = plots.plot_temporal_Molchan_diagram(
        probs, observations,
        name=name,
        axes=axes[1],
        show=False,
        plot_uniform=(i == 0),
        plot_args={
            'linecolor': color,
            'linestyle': linestyle,
            'title': f'Temporal Molchan Diagram (M{MIN_MAGNITUDE}+)'
        }
    )
    
    results.append({
        'Model': name,
        'AUC': auc,
        'ASS': ass,
        'ASS_std': sigma
    })
    print(f"  {name}: AUC={auc:.3f}, ASS={ass:.3f}±{sigma:.3f}")

plt.tight_layout()
output_path = os.path.join(script_dir, 'temporal_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# Results summary
# =============================================================================

print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)
print(f"Period: {START_TIME} to {end_dt.strftime('%Y-%m-%d')} ({n_days} days)")
print(f"Events: {n_events} M{MIN_MAGNITUDE}+")
print()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AUC', ascending=False)
print(results_df.to_string(index=False))

print(f"\nPlot saved: {output_path}")
print("="*70)

# Save results to CSV
results_path = os.path.join(script_dir, 'temporal_comparison_results.csv')
results_df.to_csv(results_path, index=False)
print(f"Results saved: {results_path}")
