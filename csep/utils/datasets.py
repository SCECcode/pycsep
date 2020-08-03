import os


_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
helmstetter_mainshock_fname = os.path.join(_root_dir, 'artifacts', 'ExampleForecasts', 'GriddedForecasts', 'helmstetter_et_al.hkj-fromXML.dat')
helmstetter_aftershock_fname = os.path.join(_root_dir, 'artifacts', 'ExampleForecasts', 'GriddedForecasts', 'helmstetter_et_al.hkj.aftershock-fromXML.dat')