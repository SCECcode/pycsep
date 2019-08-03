import os
import dask.dataframe as dd
from csep import load_stochastic_event_set

def main():
    root_dir = '/Users/wsavran/Desktop/working/ucerf3_ridgecrest_eq'
    filename = os.path.join(root_dir, 'searles_valley_m71_finite_flt', 'results_complete.bin')
    u3catalogs = load_stochastic_event_set(filename=filename, type='ucerf3', format='native', name='UCERF3-ETAS')
    for catalog in u3catalogs:
        df = catalog.get_dataframe()
        ddf = dd.from_pandas(df)
        dd.to_parquet(ddf, '/Users/wsavran/Desktop/test')
        break

if __name__ == "__main__":
    main()