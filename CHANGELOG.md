# v0.4.2 (11/03/2021)

## Change-log
- Removed normalization of rates on CL-Test (#117)
- Added function to compute bin-wise log-likelihood scores (#118)
- Properly use region to compute spatial counts and spatial magnitude counts in region class (#122)
- Fix for simplified 'fast' lat-lon ratio calculation (#125)
- Add feature to read forecasts with swapped lat/lon values in file (#130)
- Add 'percentile' argument for plotting Poisson evaluations (#131)
- Modify comparison plot to simultaneously plot T and W tests (#132)
- Add feature to trace region outline when plotting spatial data sets (#133)
- Better handling for magnitude ticks in plotting catalogs (#134)
- Refactor polygon to models module (#135)
- Added arguments to modify the fontsize for grid labels and basemap plots (#136)
- Added function to return the midpoints of the valid testing region (#137)
- Include regions when serializing catalogs to JSON (#138)
- Add support for spatial forecasts (#142)
- Upated CI workflows to reduce time required and fix intermittent OOM issues (#145)
- Adds function `get_event_counts` to catalog forecasts (#146)
- Updated README.md (#147)

## Credits
Jose Bayona (@bayonato89)
Marcus Hermann (@mherrmann3)
Pablo Iturrieta (@pabloitu)
Philip Maechling (@pjmaechling)


# v0.4.1 04/14/2021
    - Added 'fast' projection option for plotting spatial datasets (#110)
    - Fixed region border missing when plotted in various projections (#110)
    - Fixed bug where ascii catalog-based forecasts could be incorrectly loaded (#111)

# v0.4.0 03/24/2021 
    - Fixed issue in plot_poisson_consistency_test where one_sided_lower argument not coloring markers correctly
    - Added several plot configurations based on Cartopy 
      - Plotting spatial datasets with ESRI basemap
      - Plotting catalog
      - Plotting regions using outline of polygon
      - Added defaults to forecasts and catalogs
    - Added reader for gzipped UCERF3-ETAS forecasts
    - Updates for INGV readers
    - Fixed bug causing certain events to be placed into incorrrect bins
      
# v0.2 11/11/2020
  Added new catalog formats, support for masked forecast bins, and bug fixes, where applicable PR id are shown in parenthesis.

    - Fixed bug where filtering by catalog by lists did not remove all desired events (#37)
    - Included fast reader for Horus catalog (#39)
    - Modified INGV emrcmt reader (#40)
    - Fixed ndk reader and added unit tests (#44)
    - Fixed issue where magnitues were not correctly bound to gridded-forecast class (#46)
    - Fixed issue where forecasts did not work if lat/lon were not sorted (#47)
    - Fixed minor bug where catalog class did not implement inherited method (#52)
    - Gridded forecasts now parse flag from the ASCII file (#50)
    - Fixed issue where catalog did not filter properly using datetimes (#55)
    


# v0.1 10/08/2020
    Initial release to PyPI and conda-forge

    - Poisson evaluations for gridded forecasts
    - Likelihood-free evaluations for catalog-based forecasts
    - Catalog gridding and filtering
    - Plotting utilities
    - Forecast input and output
    - Documentation at docs.cseptesting.org

# v0.1-dev, 08/16/2018 -- Initial release.
