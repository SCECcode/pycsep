import os

# useful module level variables
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_gridded_forecast_root = os.path.join(_root_dir, 'artifacts', 'ExampleForecasts', 'GriddedForecasts')
_catalog_forecast_root = os.path.join(_root_dir, 'artifacts', 'ExampleForecasts', 'CatalogForecasts')
_observed_catalog_root = os.path.join(_root_dir, 'artifacts', 'ObservedCatalogs')
_polygon_region_root = os.path.join(_root_dir, 'artifacts', 'Regions', 'Polygons')

# filename to gridded forecast files
helmstetter_mainshock_fname = os.path.join(_gridded_forecast_root, 'helmstetter_et_al.hkj-fromXML.dat')
helmstetter_aftershock_fname = os.path.join(_gridded_forecast_root, 'helmstetter_et_al.hkj.aftershock-fromXML.dat')

# filename to data forecast file
ucerf3_ascii_format_landers_fname = os.path.join(_catalog_forecast_root, 'ucerf3-landers_1992-06-28T11-57-34-14.csv')

# observed catalog file name
comcat_example_catalog_fname = os.path.join(_observed_catalog_root, 'sample_comcat_catalog.csv')

# relm region polygon filenames
relm_testing_polygon_fname = os.path.join(
    _polygon_region_root, 'California', 'RELMTestingPolygon.txt'
)
relm_collection_polygon_fname = os.path.join(
    _polygon_region_root, 'California', 'RELMCollectionPolygon.txt'
)

italy_testing_polygon_fname = os.path.join(
    _polygon_region_root, 'Italy', 'ItalyTestingPolygon.txt'
)
italy_collection_polygon_fname = os.path.join(
    _polygon_region_root, 'Italy', 'ItalyCollectionPolygon.txt'
)