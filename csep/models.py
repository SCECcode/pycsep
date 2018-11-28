"""
Right now, general classes used in the CSEP model. Later, will be used to connect with database backend.
"""
import csep

class Simulation:
    def __init__(self, filename=None, min_mw=None, start_time=None, name=None, sim_type=None):
        self.filename = filename
        self.min_mw = min_mw
        self.start_time = start_time
        self.name = name
        self.sim_type = sim_type
        self.catalogs = None

        if filename is not None:
            self.catalogs = csep.load_stochastic_event_set(type=self.sim_type,
                                                           filename=self.filename,
                                                           name=self.name)

    def __str__(self):
        return 'Name: {}\n\nFilename: {}\nStart Time: {}\nMin Mw: {}\nType: {}'.format(self.name,
                                                                                       self.filename,
                                                                                       self.start_time,
                                                                                       self.min_mw,
                                                                                       self.sim_type)
