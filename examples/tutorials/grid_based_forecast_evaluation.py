"""
Grid-based Forecast Evaluation
==============================

This example demonstrates how to evaluate a grid-based and time-independent forecast.

Overview:
    1. Define forecast properties (time horizon, spatial region, etc).
    2. Obtain evaluation catalog
    3. Apply Poissonian evaluations for grid-based forecasts
    4. Store evaluation results using JSON format
    5. Visualize evaluation results
"""

####################################################################################################################################
# Load required libraries
# -----------------------
#
# Most of the core functionality can be imported from the top-level :mod:`csep` package. Utilities are available from the
# :mod:`csep.utils` subpackage.

import csep

####################################################################################################################################
# Define forecast properties
# --------------------------
#