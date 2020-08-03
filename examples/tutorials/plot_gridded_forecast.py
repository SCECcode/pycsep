"""
Plotting gridded forecast
=========================

This example show you how to load a gridded forecast stored in the default ASCII format.
"""

####################################################################################################################################
# Load required libraries
# -----------------------
#
# Most of the core functionality can be imported from the top-level :mod:`csep` package. Utilities are available from the
# :mod:`csep.utils` subpackage.

import csep
from csep.utils import datasets

####################################################################################################################################
# Load forecast
# -------------
#
# For this example, we provide the example forecast data set along with the main repository. The filepath is relative
# to the root directory of the package. You can specify any file location for your forecasts.

forecast = csep.load_gridded_forecast(datasets.helmstetter_mainshock_fname)

####################################################################################################################################
# Plot forecast
# -------------
#
# The forecast object provides :meth:`csep.core.forecasts.GriddedForecast.plot` to plot a gridded forecast. This function
# returns a matplotlib axes, so more specific attributes can be set on the figure.

ax = forecast.plot()



