.. _alarm-forecast-tutorial:

Alarm-Based Forecast Evaluation
===============================

This tutorial demonstrates how to load, evaluate, and visualize alarm-based earthquake forecasts
using pyCSEP. Alarm-based forecasts assign a score (e.g., probability or alarm level) to each
spatial cell rather than providing absolute earthquake rates.

.. contents:: Table of Contents
    :local:
    :depth: 2

Overview
--------

Alarm-based forecasts are evaluated using:

- **ROC (Receiver Operating Characteristic) curves**: Plot hit rate vs false alarm rate
- **Molchan diagrams**: Plot miss rate vs alarm fraction

These methods assess how well the forecast ranks spatial cells by their probability of experiencing
an earthquake.

Loading an Alarm Forecast
-------------------------

First, let's load an alarm forecast from a CSV file:

.. code-block:: python

    import csep
    from csep.utils import plots

    # Load alarm forecast
    forecast = csep.load_alarm_forecast(
        'alarm_forecast.csv',
        name='MyAlarmForecast',
        score_field='alarm_score'  # Column to use as score
    )

    print(f"Loaded forecast: {forecast.name}")
    print(f"Spatial cells: {forecast.region.num_nodes}")
    print(f"Magnitude range: {forecast.magnitudes[0]:.1f} - {forecast.magnitudes[-1]:.1f}")

The CSV file should have at minimum longitude, latitude, and a score column:

.. code-block:: text

    lon,lat,alarm_score,magnitude_min,magnitude_target
    165.0,-47.0,0.234,4.0,7.0
    165.1,-47.0,0.156,4.0,7.0
    165.2,-47.0,0.089,4.0,7.0

Querying an Observed Catalog
----------------------------

Next, query the observed earthquake catalog for the forecast period:

.. code-block:: python

    import datetime

    # Query GeoNet catalog for New Zealand
    catalog = csep.query_gns(
        start_time=datetime.datetime(2020, 1, 1),
        end_time=datetime.datetime(2021, 1, 1),
        min_magnitude=4.0,
        min_latitude=-47.0,
        max_latitude=-34.0,
        min_longitude=165.0,
        max_longitude=180.0
    )

    print(f"Found {catalog.event_count} events")

    # Filter catalog to match forecast region (optional)
    catalog_filtered = catalog.filter_spatial(region=forecast.region)

Evaluating with ROC Curve
-------------------------

The ROC curve shows the trade-off between hit rate (true positive rate) and
false alarm rate (false positive rate) at different alarm thresholds:

.. code-block:: python

    # Create ROC diagram
    ax, auc = plots.plot_ROC_diagram(
        forecast, catalog_filtered,
        linear=True,  # Use linear scale
        plot_args={
            'title': 'ROC Curve - My Alarm Forecast',
            'linecolor': 'blue'
        }
    )

    print(f"Area Under Curve (AUC): {auc:.3f}")

**Interpretation:**

- **AUC = 1.0**: Perfect forecast (upper-left corner)
- **AUC = 0.5**: Random forecast (diagonal line)
- **AUC < 0.5**: Worse than random (anti-skill)

Evaluating with Molchan Diagram
-------------------------------

The Molchan diagram emphasizes the trade-off between capturing earthquakes
and the fraction of space occupied by alarms:

.. code-block:: python

    # Create Molchan diagram
    ax, ass, sigma = plots.plot_Molchan_diagram(
        forecast, catalog_filtered,
        linear=True,
        plot_args={
            'title': 'Molchan Diagram - My Alarm Forecast',
            'linecolor': 'red'
        }
    )

    print(f"Area Skill Score (ASS): {ass:.3f} ± {sigma:.3f}")

**Interpretation:**

- **ASS = 1.0**: Perfect forecast (captures all events with minimal space)
- **ASS = 0.5**: No skill (same as random)
- **ASS < 0.5**: Negative skill (worse than random)

Comparing Multiple Score Fields
-------------------------------

You can load forecasts with different score fields and compare them:

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    score_fields = ['alarm_score', 'probability', 'rate_per_day']
    colors = ['blue', 'red', 'green']

    for score_field, color in zip(score_fields, colors):
        forecast = csep.load_alarm_forecast(
            'alarm_forecast.csv',
            name=f'{score_field}',
            score_field=score_field
        )

        # ROC
        plots.plot_ROC_diagram(
            forecast, catalog_filtered,
            axes=axes[0], linear=True, show=False,
            plot_args={'linecolor': color}
        )

        # Molchan
        plots.plot_Molchan_diagram(
            forecast, catalog_filtered,
            axes=axes[1], linear=True, show=False,
            plot_args={'linecolor': color}
        )

    plt.tight_layout()
    plt.savefig('multi_score_comparison.png', dpi=150)

Complete Example
----------------

Here's a complete working example:

.. code-block:: python

    #!/usr/bin/env python
    """Complete alarm forecast evaluation example."""

    import csep
    from csep.utils import plots
    import matplotlib.pyplot as plt
    import datetime

    # 1. Load alarm forecast
    forecast = csep.load_alarm_forecast(
        'nz_alarm_forecast.csv',
        name='NZ_Alarm_Model'
    )

    # 2. Query observed catalog
    catalog = csep.query_gns(
        start_time=datetime.datetime(2024, 1, 1),
        end_time=datetime.datetime(2025, 1, 1),
        min_magnitude=4.0,
        min_latitude=-47.0,
        max_latitude=-34.0,
        min_longitude=165.0,
        max_longitude=179.0
    )

    # 3. Filter to forecast region
    catalog_filtered = catalog.filter_spatial(region=forecast.region)

    # 4. Create evaluation plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1, auc = plots.plot_ROC_diagram(
        forecast, catalog_filtered,
        axes=axes[0], linear=True, show=False
    )

    ax2, ass, sigma = plots.plot_Molchan_diagram(
        forecast, catalog_filtered,
        axes=axes[1], linear=True, show=False
    )

    plt.tight_layout()
    plt.savefig('alarm_evaluation.png', dpi=150)

    # 5. Print results
    print(f"AUC: {auc:.3f}")
    print(f"ASS: {ass:.3f} ± {sigma:.3f}")

See Also
--------

- :ref:`Forecasts <forecast-reference>` - Alarm-based forecast format
- :ref:`Evaluations <evaluation-reference>` - ROC and Molchan evaluation details
- :func:`csep.load_alarm_forecast` - API reference
- :func:`csep.utils.plots.plot_ROC_diagram` - ROC diagram API
- :func:`csep.utils.plots.plot_Molchan_diagram` - Molchan diagram API
