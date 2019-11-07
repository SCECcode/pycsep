import time
from csep import load_comcat, load_stochastic_event_sets
from csep.utils.time import epoch_time_to_utc_datetime
from csep.utils.constants import SECONDS_PER_ASTRONOMICAL_YEAR


def do_evaluate_catalogs(filename, task, task_configuration):
    """ wrapper for individual catalog processing task. will parallelize over these
        using dask delayed or python multiprocessing.
    """
    region = task_configuration.get('region', None)
    sim_type = task_configuration.get('ucerf3', 'ucerf3')
    name = task_configuration.get('name', 'UCERF3-ETAS')
    # these are required
    start_epoch = task_configuration['start_epoch']
    end_epoch = task_configuration['end_epoch']
    n_cat = task_configuration['n_cat']
    plot_dir = task_configuration['plot_dir']

    # compute needed stuff
    time_horizon = (end_epoch - start_epoch) / SECONDS_PER_ASTRONOMICAL_YEAR / 1000

    # get name of task
    task_name = task.__class__.__name__

    # download comcat catalog
    print(f'{task_name}: Downloading Comcat catalog.')
    try:
        comcat = load_comcat(epoch_time_to_utc_datetime(start_epoch), epoch_time_to_utc_datetime(end_epoch),
                             min_magnitude=2.50,
                             min_latitude=31.50, max_latitude=43.00,
                             min_longitude=-125.40, max_longitude=-113.10)
    # if it times out, try again
    except:
        comcat = load_comcat(epoch_time_to_utc_datetime(start_epoch), epoch_time_to_utc_datetime(end_epoch),
                             min_magnitude=2.50,
                             min_latitude=31.50, max_latitude=43.00,
                             min_longitude=-125.40, max_longitude=-113.10)
    comcat = comcat.filter_spatial(region)
    print(comcat)

    # get iterator for catalog files
    u3 = load_stochastic_event_sets(filename=filename, type=sim_type, name=name, region=region)

    # first pass through catalogs
    t0 = time.time()
    for i, cat in enumerate(u3):
        if region:
            cat_filt = cat.filter(f'origin_time < {end_epoch}').filter_spatial(region)
        else:
            cat_filt = cat.filter(f'origin_time < {end_epoch}')

        # don't need to copy here, so long as mw are not decreasing in task definition
        task.process_catalog(cat_filt)

        # breaking for development
        if (i + 1) % n_cat == 0:
            break

        if (i+1) % 1000 == 0:
            t1 = time.time()
            print(f'{task_name}: Processed {i+1} catalogs in {t1-t0} seconds')


    output = {}
    t0 = time.time()
    # get iterator for catalog files for second pass. most tasks do not require this and will simply pass it
    if task.needs_two_passes:
        u3 = load_stochastic_event_sets(filename=filename, type=sim_type, name=name, region=region)
        for i, cat in enumerate(u3):
            cat_filt = cat.filter(f'origin_time < {end_epoch}').filter_spatial(region)
            task.process_again(cat_filt, args=(time_horizon, n_cat, end_epoch, comcat))

            # breaking for development
            if (i + 1) % n_cat == 0:
                break

            if (i + 1) % 1000 == 0:
                t1 = time.time()
                print(f'{task_name}: Processed {i+1} catalogs in {t1-t0} seconds')


    # compute evaluation and plot
    result = task.post_process(comcat, args=(None, time_horizon, end_epoch, n_cat))

    # plot task
    task.plot(result, plot_dir, show=False)

    # compile results and return
    output[task_name] = result
    return output