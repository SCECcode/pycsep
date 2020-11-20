import csep
from csep.utils import datasets
import cartopy

## Examples
# California Forecast

forecast = csep.load_gridded_forecast(datasets.helmstetter_mainshock_fname,
                                      name='Helmstetter et al (2007) California')
ax = forecast.plot(plot_args={'figsize': (8,8), 'title_size': 18,
                              'coastline':True, 'linecolor':'gray',
                              'basemap':'ESRI_terrain',
                              'alpha_exp': 0.5}, show=True)

# Italy Forecast
forecast = csep.load_gridded_forecast(datasets.hires_ssm_italy_fname,
                                      name='Werner, et al (2010) Italy')
ax = forecast.plot(extent=[1, 23, 28, 51],
                   show=True,
                   plot_args={'title': 'Italy 10 year forecast',
                              'borders':True, 'feature_lw': 0.5,
                              'basemap':'ESRI_relief',
                              'cmap': 'jet',
                              'alpha_exp': 0.8,
                              'projection': cartopy.crs.EuroPP()})

# Global Forecast
forecast = csep.load_gridded_forecast(datasets.gear1_downsampled_fname,
                                      name='GEAR1 Forecast (downsampled)')
ax = forecast.plot(set_global=True, show=True,
                   plot_args={'figsize': (10,6), 'coastline':True, 'feature_color':'black',
                              'projection': cartopy.crs.Robinson(central_longitude=-180.0),
                              'cmap': 'magma', 'grid_labels': False})
