import os


_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# filename to forecast files
helmstetter_mainshock_fname = os.path.join(_root_dir, 'artifacts', 'ExampleForecasts', 'GriddedForecasts', 'helmstetter_et_al.hkj-fromXML.dat')
helmstetter_aftershock_fname = os.path.join(_root_dir, 'artifacts', 'ExampleForecasts', 'GriddedForecasts', 'helmstetter_et_al.hkj.aftershock-fromXML.dat')
ucerf3_ascii_format_landers_fname = os.path.join(_root_dir, 'artifacts', 'ExampleForecasts', 'CatalogForecasts', 'ucerf3-landers_1992-06-28T11-57-34-14.csv')