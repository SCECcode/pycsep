# v0.6.3 (2/1/2024)

# Change-log

Added test for Winodws 10 on GitHub actions ([#244](https://github.com/SCECcode/pycsep/pull/244))  
Removed shading in plotting fewer than 3 forecasts ([#247](https://github.com/SCECcode/pycsep/pull/247))  
Fixed tutorial for plot_customizations ([#242](https://github.com/SCECcode/pycsep/pull/242))  
Fixed negative binomial consistency plots now have the correct boundaries ([#245](https://github.com/SCECcode/pycsep/pull/245))  
Changed environment build of pypi-publish from miniconda to micromamba ([#238](https://github.com/SCECcode/pycsep/pull/238))  
Fixed negative timestamps parsing for Windows ([#230](https://github.com/SCECcode/pycsep/pull/230))  
Fixed writing catalog csv files on Windows ([#235](https://github.com/SCECcode/pycsep/pull/235))

## Credits
Pablo Iturrieta (@pabloitu)  
William Savran (@wsavran)  
Philip Maechling (@pjmaechling)  
Fabio Silva (@fabiolsilva)

# v0.6.2 (6/16/2023)

# Change-log
Fixed an error-bar bug for normalized consistency plots ([#222](https://github.com/SCECcode/pycsep/pull/222))  
Fixed handles URL exception or SSL verifications errors for both python 3.8 and 3.11 inclusive ([#231](https://github.com/SCECcode/pycsep/pull/231))  
Included CMT Catalog accessor ([#217](https://github.com/SCECcode/pycsep/pull/217))  
Added NZ catalog reader ([#213](https://github.com/SCECcode/pycsep/pull/213))

## Credits
Pablo Iturrieta (@pabloitu)  
Kenny Graham (@KennyGraham1)  
Fabio Silva (@fabiolsilva)

# v0.6.1 (12/12/2022)

# Change-log
Added quadtree csv reader ([#186](https://github.com/SCECcode/pycsep/pull/186))  
Non-Poissonian tests
([#189](https://github.com/SCECcode/pycsep/pull/189),
[#205](https://github.com/SCECcode/pycsep/pull/205),
[#208](https://github.com/SCECcode/pycsep/pull/208),
[#209](https://github.com/SCECcode/pycsep/pull/209))  
Added plots for p-values, and confidence ranges for consistency tests ([#190](https://github.com/SCECcode/pycsep/pull/190))    
Added NZ testing and collection regions ([#198](https://github.com/SCECcode/pycsep/pull/198))  
Fixed region border plotting issue ([#199](https://github.com/SCECcode/pycsep/pull/199))  
Added documentation for non-Poissonian tests ([#202](https://github.com/SCECcode/pycsep/pull/202))  
Support for BSI catalog ([#201](https://github.com/SCECcode/pycsep/pull/201))  
Fixed compatibility with new version of matplotlib ([#206](https://github.com/SCECcode/pycsep/pull/206))

## Credits
Pablo Iturrieta (@pabloitu)  
Jose Bayona (@bayonato89)  
Khawaja Asim (@khawajasim)  
William Savran (@wsavran)

# v0.6.0 (02/04/2022)

## Change-log
Adds support for quadtree regions ([#184](https://github.com/SCECcode/pycsep/pull/184))

## Credits
Khawaja Asim (@khawajasim)  
William Savran (@wsavran)

# v0.5.2 (01/25/2022)
## Change-log
Fixed failing build from matplotlib 3.5.0 release (#162)  
Updates to documentation and tutorials (#165)  
Added theory of tests to documentation (#171)  
Added notebooks folder for community recipes (#173)  
Pin numpy version to 1.25.1 to fix (#168) 

## Credits
William Savran (@wsavran)
Kirsty Bayliss (@kirstybayliss)

# v0.5.1 (11/15/2021)

## Change-log
Modified plot_spatial_dataset and plot_catalog to correctly handle Global plots (#150)  
Updated plot_customizations example in the docs (#150)  
Added DOI badge and conda downloads badge (#156)  
Added plotting args to catalog evaluation plots (#154)  
Add option to ignore colorbar in spatial dataset plots (#154)

## Credits
William Savran (@wsavran)  
Pablo Iturrieta (@pabloitu)

# v0.5.0 (11/03/2021)

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
