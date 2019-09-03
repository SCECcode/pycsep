import matplotlib

import seaborn as sns

from csep.core.analysis import ucerf3_consistency_testing
from csep.utils.time import utc_now_epoch


matplotlib.use('agg')
matplotlib.rcParams['figure.max_open_warning'] = 150
sns.set()


if __name__ == "__main__":
    sim_dir = '/Users/wsavran/Desktop/working/ucerf3_ridgecrest_eq/searles_valley_m71_shakemap_src'
    event_id = 'ci38457511'
    end_epoch = utc_now_epoch()
    ucerf3_consistency_testing(sim_dir, event_id, end_epoch, n_cat=25000)
