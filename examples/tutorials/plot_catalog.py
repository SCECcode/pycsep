import csep
from csep.utils import time_utils


min_mw = 2.95

start_time = time_utils.strptime_to_utc_datetime("2008-01-01 00:00:00")
end_time = time_utils.strptime_to_utc_datetime("2010-01-01 00:00:00")

comcat_catalog = csep.query_comcat(start_time, end_time, min_magnitude=min_mw)

ax2 = comcat_catalog.plot(show=True)