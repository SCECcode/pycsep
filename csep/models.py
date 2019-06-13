class Simulation:
    """
    View of CSEP Experiment. Contains minimal information required to perform evaluations of
    CSEP Forecasts
    """
    def __init__(self, filename='', min_mw=2.5, start_time=-1, sim_type='', name=''):
        self.filename = filename
        self.min_mw = min_mw
        self.start_time = start_time
        self.sim_type = sim_type
        self.name = name
