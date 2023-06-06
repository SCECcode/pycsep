import csep
import numpy
from csep.core.brier_evaluations import brier_score_test
from csep.utils.plots import plot_consistency_test
from datetime import datetime

forecast = csep.load_gridded_forecast(csep.datasets.hires_ssm_italy_fname)

start = datetime(2010, 1, 1)
end = datetime(2015, 1, 1)
cat = csep.query_bsi(start, end, min_magnitude=5.0)
cat.filter_spatial(region=csep.regions.italy_csep_region(
    magnitudes=numpy.arange(5.0, 9.0, 0.1)), in_place=True)

result = brier_score_test(forecast, cat)
plot_consistency_test(result, show=True)