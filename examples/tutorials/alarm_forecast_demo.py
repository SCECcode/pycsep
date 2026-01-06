#!/usr/bin/env python
"""
Demonstration of Alarm-Based Forecast Evaluation in pyCSEP

This script demonstrates how to:
1. Load an alarm-based earthquake forecast from CSV
2. Query a real observed catalog from GeoNet (New Zealand)
3. Evaluate the forecast using ROC curves and Molchan diagrams

This implementation works with most geographic regions worldwide.
"""

import sys
import os

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import csep
from csep.utils import plots
from csep.core.catalogs import CSEPCatalog
import datetime

def create_nz_alarm_forecast(filename='nz_alarm_forecast.csv', resolution=0.1):
    """Create a realistic alarm forecast for New Zealand region."""
    print("Creating New Zealand alarm forecast...")

    # New Zealand region bounds
    min_lon, max_lon = 165.0, 179.0
    min_lat, max_lat = -47.0, -34.0

    # Create grid of cells
    lons = np.arange(min_lon, max_lon, resolution)
    lats = np.arange(min_lat, max_lat, resolution)

    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()

    # Generate realistic alarm scores based on proximity to known seismic zones
    # Higher scores near the Alpine Fault and Hikurangi subduction zone
    scores = np.zeros(len(lon_flat))

    for i, (lon, lat) in enumerate(zip(lon_flat, lat_flat)):
        # Alpine Fault (South Island) - runs roughly NE-SW
        alpine_dist = abs((lon - 170.0) + 0.5 * (lat + 43.0))
        alpine_score = np.exp(-alpine_dist**2 / 2.0)

        # Hikurangi subduction zone (North Island east coast)
        hikurangi_dist = np.sqrt((lon - 178.0)**2 + (lat + 39.0)**2)
        hikurangi_score = np.exp(-hikurangi_dist**2 / 4.0)

        # Taupo Volcanic Zone (North Island)
        taupo_dist = np.sqrt((lon - 176.0)**2 + (lat + 38.5)**2)
        taupo_score = np.exp(-taupo_dist**2 / 1.5)

        # Combine scores
        scores[i] = max(alpine_score, hikurangi_score, taupo_score)

    # Normalize to [0, 1]
    scores = scores / scores.max()

    # Add some noise to make it more realistic
    scores += np.random.normal(0, 0.05, len(scores))
    scores = np.clip(scores, 0, 1)

    # Create CSV data
    data = []
    for i in range(len(lon_flat)):
        row = {
            'lon': lon_flat[i],
            'lat': lat_flat[i],
            'alarm_score': scores[i],
            'probability': scores[i] * 0.85,  # Slightly different
            'rate_per_day': scores[i] * 0.0005,  # Events per day
            'magnitude_min': 4.0,
            'magnitude_target': 7.0
        }
        data.append(row)

    # Write to CSV
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lon', 'lat', 'alarm_score',
                                               'probability', 'rate_per_day',
                                               'magnitude_min', 'magnitude_target'])
        writer.writeheader()
        writer.writerows(data)

    print(f"Created {filename} with {len(data)} spatial cells")
    print(f"  Region: New Zealand ({min_lon}°E to {max_lon}°E, {min_lat}°N to {max_lat}°N)")
    return filename


def query_real_catalog(region='new_zealand'):
    """Query real earthquake catalog from GeoNet (New Zealand)."""
    print(f"\nQuerying real earthquake catalog for {region}...")

    # Define time window (e.g., last year)
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=365)

    print(f"  Time window: {start_time.date()} to {end_time.date()}")

    try:
        # Query GeoNet catalog for New Zealand
        catalog = csep.query_gns(
            start_time=start_time,
            end_time=end_time,
            min_magnitude=4.0,
            min_latitude=-47.0,
            max_latitude=-34.0,
            min_longitude=165.0,
            max_longitude=179.0,
            max_depth=100.0
        )

        print(f"  Successfully downloaded {catalog.event_count} events from GeoNet")
        print(f"  Magnitude range: {catalog.min_magnitude:.1f} - {catalog.max_magnitude:.1f}")

        return catalog

    except Exception as e:
        print(f"  Warning: Could not query GeoNet catalog: {e}")
        print(f"  Falling back to synthetic catalog...")
        return create_synthetic_catalog_nz()


def create_synthetic_catalog_nz(n_events=100):
    """Create synthetic catalog for New Zealand if real data unavailable."""
    print(f"  Creating synthetic catalog with {n_events} events...")

    # New Zealand region bounds
    min_lon, max_lon = 165.0, 179.0
    min_lat, max_lat = -47.0, -34.0

    import time
    current_time = int(time.time() * 1000)

    events = []
    for i in range(n_events):
        # Bias events toward known seismic zones
        if np.random.random() < 0.4:  # Alpine Fault region
            lon = np.random.uniform(169.0, 171.0)
            lat = np.random.uniform(-44.0, -42.0)
        elif np.random.random() < 0.7:  # Hikurangi/North Island
            lon = np.random.uniform(176.0, 178.5)
            lat = np.random.uniform(-41.0, -37.0)
        else:  # Other regions
            lon = np.random.uniform(min_lon, max_lon)
            lat = np.random.uniform(min_lat, max_lat)

        # Realistic magnitude distribution (Gutenberg-Richter)
        magnitude = 4.0 - np.log10(np.random.random()) / 1.0
        magnitude = min(magnitude, 7.0)

        depth = np.random.uniform(5, 40)  # Realistic depth range

        event = (i, current_time + i*1000, float(lat), float(lon),
                float(depth), float(magnitude))
        events.append(event)

    catalog = CSEPCatalog(data=events)
    print(f"  Created synthetic catalog with {catalog.event_count} events")
    return catalog


def evaluate_alarm_forecast(forecast, catalog, output_dir='.'):
    """Evaluate alarm forecast using ROC and Molchan diagrams."""
    print("\nEvaluating alarm forecast...")

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Molchan diagram (left)
    print("  - Creating Molchan diagram...")
    plots.plot_Molchan_diagram(
        forecast, catalog,
        linear=True,
        axes=axes[0],
        show=False,
        savepdf=False,
        savepng=False
    )
    axes[0].set_title('Molchan Diagram\n(Miss Rate vs Alarm Fraction)', fontsize=12)

    # ROC curve (right)
    print("  - Creating ROC diagram...")
    plots.plot_ROC_diagram(
        forecast, catalog,
        linear=True,
        axes=axes[1],
        show=False,
        savepdf=False,
        savepng=False
    )
    axes[1].set_title('ROC Curve\n(Hit Rate vs False Alarm Rate)', fontsize=12)

    plt.tight_layout()

    # Save figure
    output_file = f'{output_dir}/nz_alarm_evaluation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved evaluation plots to: {output_file}")
    plt.close()


def main():
    """Main demonstration workflow."""
    print("="*70)
    print("Alarm-Based Forecast Evaluation Demo - New Zealand")
    print("="*70)

    # Step 1: Create New Zealand alarm forecast
    forecast_file = create_nz_alarm_forecast(resolution=0.1)

    # Step 2: Load alarm forecast
    print("\nLoading alarm forecast...")
    forecast = csep.load_alarm_forecast(
        forecast_file,
        name='NZ_ALARM_Model',
        score_field='alarm_score'
    )
    print(f"Loaded forecast: {forecast.name}")
    print(f"  - Spatial cells: {forecast.region.num_nodes}")
    print(f"  - Magnitude range: {forecast.magnitudes[0]:.1f} - {forecast.magnitudes[-1]:.1f}")

    # Step 3: Query real catalog from GeoNet
    catalog = query_real_catalog(region='new_zealand')

    # Step 4: Filter catalog to match forecast region
    print(f"\nFiltering catalog to forecast region...")
    try:
        catalog_filtered = catalog.filter_spatial(region=forecast.region)
        print(f"  Filtered catalog: {catalog_filtered.event_count} events in forecast region")
        catalog_to_use = catalog_filtered if catalog_filtered.event_count > 0 else catalog
    except:
        print(f"  Using full catalog: {catalog.event_count} events")
        catalog_to_use = catalog

    # Step 5: Evaluate forecast
    print(f"\nEvaluating forecast with {catalog_to_use.event_count} observed events...")
    evaluate_alarm_forecast(forecast, catalog_to_use)

    # Step 6: Demonstrate different score fields
    print("\n" + "="*70)
    print("Comparing Different Score Fields")
    print("="*70)

    score_fields = ['alarm_score', 'probability', 'rate_per_day']

    for score_field in score_fields:
        print(f"\nScore field: '{score_field}'")
        forecast_variant = csep.load_alarm_forecast(
            forecast_file,
            name=f'NZ_{score_field}',
            score_field=score_field
        )

        print(f"  Forecast: {forecast_variant.name}")
        print(f"  Score range: [{forecast_variant.data.min():.6f}, {forecast_variant.data.max():.6f}]")
        print(f"  Mean score: {forecast_variant.data.mean():.6f}")

    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print("\nKey files created:")
    print(f"  - {forecast_file}")
    print(f"  - nz_alarm_evaluation.png")
    print("\nThis demo shows:")
    print("  ✓ Loading alarm forecasts for any region (New Zealand example)")
    print("  ✓ Querying real earthquake catalogs from GeoNet")
    print("  ✓ Evaluating forecasts with ROC curves and Molchan diagrams")
    print("  ✓ Comparing different score fields (alarm_score, probability, rate_per_day)")


if __name__ == '__main__':
    main()
