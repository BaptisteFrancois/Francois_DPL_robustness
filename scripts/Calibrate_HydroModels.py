import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import random
from deap import base, creator, tools, algorithms, tools
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from HydroModels import run_abcd
import HydroModels
from validation_models import plot_validation_figures
from multiprocessing import Pool
# import inspect for DEBUG  # print(inspect.getsource(HydroModels.run_abcd))

# force‐reload to pick up your edits
import importlib
importlib.reload(HydroModels)


random.seed(42)
np.random.seed(42)

def evaluate(obs_flow, prcp, tmin, tmax, day_of_year, lat, flowlen, lows, ups, norm_indiv):
    
    # 1) convert normalized [0,1] → real values
    norm = np.asarray(norm_indiv, dtype=float)
    real_params = lows + norm * (ups - lows)

    #print(f"Evaluating parameters: {real_params}")
    
    # 1) cast your twelve GA genes into named scalars
    gene_names = [
        'm','rain_thr','snow_thr',
        'pet1','pet2','pet3',
        'a','b','c','d',
        'N','K','VELO','DIFF'
    ]
    vals = list(real_params)
    params = dict(zip(gene_names, vals))
    # then pull them out:
    m         = float(params['m'])
    rain_thr  = float(params['rain_thr'])
    snow_thr  = float(params['snow_thr'])
    pet1      = float(params['pet1'])
    pet2      = float(params['pet2'])   
    pet3      = float(params['pet3'])
    a         = float(params['a'])
    b         = float(params['b'])
    c         = float(params['c'])
    d         = float(params['d'])
    N         = float(params['N'])
    K         = float(params['K'])
    VELO      = float(params['VELO'])
    DIFF      = float(params['DIFF'])

    # hard‐clamp to legal ranges
    a    = max(a,   1e-6)

    N    = max(N,   1.0)
    K    = max(K,   1e-6)
    VELO = max(VELO,1e-6)
    DIFF = max(DIFF,1e-6)

    # 2) package into the exact names run_abcd expects
    snow_par    = [m, rain_thr, snow_thr]
    pet_par     = None          # still unused
    abcd_par    = [a, b, c, d]
    routing_par = [N, K, VELO, DIFF]

    # 3) CALL WITH KEYWORDS so nothing ever shifts
    sim_q, routed_d, routed_b, direct, base, pet, snow, rain, melt, uz, lz, evap = \
        run_abcd(
            prcp       = prcp,
            tmin       = tmin,
            tmax       = tmax,
            day_of_year= day_of_year,
            lat        = float(lat),
            flowlen    = flowlen,
            snow_par   = snow_par,
            pet_par    = pet_par,
            abcd_par   = abcd_par,
            routing_par= routing_par
        )

    # 4) sanity‐check shapes
    assert sim_q.shape == prcp.shape

    # 5) compute NSE
    mask = ~np.isnan(obs_flow) & ~np.isnan(sim_q)
    num  = ((obs_flow[mask] - sim_q[mask])**2).sum()
    den  = ((obs_flow[mask] - obs_flow[mask].mean())**2).sum()
    nse  = 1 - num/den

    return (-nse,)


def calibrate(obs_flow, prcp, tmin, tmax, doy, lat, flowlen, bounds,
              gage_name=None,
              log_fname=None,
              ngen=2500,
              mu=100, lamb=200,
              cxpb=0.7, mutpb=0.3,
              patience=100, min_delta=1e-5):
    
    
    """
    Run a µ+λ GA, log one row per generation into log_fname,
    and early‐stop after `patience` gens without >=min_delta improvement.
    """
    # 1) Define bounds / creator / toolbox
    lows = np.array([l for l,u in bounds])
    ups  = np.array([u for l,u in bounds])

    # at module scope, once
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)


    toolbox = base.Toolbox()
    for i,(l,u) in enumerate(bounds):
        toolbox.register(f"attr_{i}", random.random)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [getattr(toolbox,f"attr_{i}") for i in range(len(bounds))], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate",   tools.cxBlend,   alpha=0.4)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate,
                     obs_flow, prcp, tmin, tmax, doy, lat, flowlen, lows, ups)

    def clamp01(pop):
        for ind in pop:
            for i,v in enumerate(ind):
                ind[i] = min(1.0, max(0.0, v))
        return pop

    toolbox.decorate("mate",   lambda f: lambda *a,**k: clamp01(f(*a,**k)))
    toolbox.decorate("mutate", lambda f: lambda *a,**k: clamp01(f(*a,**k)))

    # 2) initialize
    pop = toolbox.population(n=mu)
    hof = tools.HallOfFame(1)
    # Evaluate parents in the starting population
    invalid0   = [ind for ind in pop if not ind.fitness.valid]
    fitnesses0 = map(toolbox.evaluate, invalid0)
    for ind, fit in zip(invalid0, fitnesses0):
        ind.fitness.values = fit

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("mean", np.mean)
    stats.register("std",  np.std)
    stats.register("min",  np.min)
    stats.register("max",  np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen","nevals"] + stats.fields

    # 3) open log file & write header
    gage = "nonamed" if gage_name is None else gage_name
    log_fname = f"calibration_log_{gage}.csv" if log_fname is None else log_fname
    pathlogs = "../results/calibration_results/logs/"
    
    log_fname = os.path.join(pathlogs, log_fname)

    with open(log_fname, "w") as flog:
        flog.write(",".join(logbook.header) + "\n")

        best = float("inf")
        no_improve = 0

        # 4) GA main loop
        for gen in range(ngen):
            # a) generate λ offspring
            offspring = algorithms.varOr(pop, toolbox, lamb, cxpb, mutpb)
            # b) evaluate invalid
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit
            # c) select next gen (μ of μ+λ)
            pop = toolbox.select(pop + offspring, mu)
            # d) update Hall of Fame
            hof.update(pop)
            # e) record stats
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid), **record)
            # f) append one line to file
            row = [ str(logbook[gen][fld]) for fld in logbook.header ]
            flog.write(",".join(row) + "\n")

            # g) early stopping?
            current = record["min"]
            if current < best - min_delta:
                best = current
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"⏱  Early stop at gen {gen} (no imp for {patience} gens)")
                break

    best_ind = hof[0]
    best_nse = -best_ind.fitness.values[0]

    # Get the parameter values of the best individual
    # The individual is normalized in the range [0, 1]
    
    # Convert the normalized individual to real values
    best_ind = np.asarray(best_ind, dtype=float)
    real_params = lows + best_ind * (ups - lows)
    
    return {"params_ga": real_params, "nse_ga": best_nse}
    


def process_gage(gage):
    
    g, patience, min_delta, bounds, cal_first, cal_fraction, paths = gage
    g = g.strip()  # remove any leading/trailing whitespace
    
    print(f"\n🌊 Calibrating gage {g}...")

    # load data
    weather_paths = os.path.join(paths['livneh'], f'livneh_lusu_basin_{g}.csv')
    obsflow_paths = os.path.join(paths['obsflow'], f'{g}_observed_flow.csv')
    shapefile_path = paths['shapefiles']
    flowlen_path = paths['flowlen']
    results_path = os.path.join(paths['calibration_results'], 'abcd_parameters_{}_CAL={}-{}.csv')
    figures_path = os.path.join(paths['calibration_results'], 'figures')


    weather = pd.read_csv(weather_paths, index_col='date', parse_dates=True)
    obs     = pd.read_csv(obsflow_paths, index_col='date', parse_dates=True)
    weather, obs = weather.align(obs, join='inner', axis=0)

    # split the time series into calibration and validation periods
    if cal_first: # the beginning of the time series is the calibration period
        cal_len = int(len(weather) * cal_fraction)
        cal_start = weather.index[0]
        cal_end = weather.index[cal_len - 1]
        val_start = weather.index[cal_len]
        val_end = weather.index[-1]
    else: # the end of the time series is the calibration period
        val_len = int(len(weather) * cal_fraction)
        val_start = weather.index[0]
        val_end = weather.index[val_len - 1]
        cal_start = weather.index[val_len]
        cal_end = weather.index[-1] 

    # Truncate the data to the calibration period
    weather_cal = weather.loc[cal_start:cal_end]
    obs_cal = obs.loc[cal_start:cal_end]

    # Arrayify the data (full period)
    prcp = weather['prcp_mm'].to_numpy(dtype=np.float64)
    tmin = weather['tmin_C'].to_numpy(dtype=np.float64)
    tmax = weather['tmax_C'].to_numpy(dtype=np.float64)
    obsf = obs['flow_mm_day'].to_numpy(dtype=np.float64)
    doy  = weather.index.day_of_year.to_numpy(dtype=np.int64)

    # Truncate the data to the calibration period
    prcp_cal = weather['prcp_mm'].loc[cal_start:cal_end].to_numpy(dtype=np.float64)
    tmin_cal = weather['tmin_C'].loc[cal_start:cal_end].to_numpy(dtype=np.float64)
    tmax_cal = weather['tmax_C'].loc[cal_start:cal_end].to_numpy(dtype=np.float64)
    obsf_cal = obs['flow_mm_day'].loc[cal_start:cal_end].to_numpy(dtype=np.float64)
    doy_cal  = weather.index.to_series().dt.day_of_year.loc[
        cal_start:cal_end].to_numpy(dtype=np.int64)

    # read static attributes
    lat_df     = gpd.read_file(shapefile_path).set_index('hru_id')
    lat_df.index = lat_df.index.astype(str).str.zfill(8)
    fl_df      = pd.read_csv(flowlen_path, index_col='hru_id')
    fl_df.index= fl_df.index.astype(str).str.zfill(8)

    lat      = float(lat_df.loc[g,'lat_cen'])
    flowlen  = float(round(fl_df.loc[g,'max_flow_length'],0))

    # run the GA‐based calibration
    results = calibrate(obsf_cal, prcp_cal, tmin_cal, tmax_cal, doy_cal, lat, flowlen, bounds=bounds,
                        gage_name=g,
                        patience=patience, 
                        min_delta=min_delta)
    
    # Write results to a CSV file
    results_df = pd.DataFrame({
        'm': results['params_ga'][0],
        'rain_thr': results['params_ga'][1],
        'snow_thr': results['params_ga'][2],
        'pet1': results['params_ga'][3],
        'pet2': results['params_ga'][4],
        'pet3': results['params_ga'][5],
        'a': results['params_ga'][6],
        'b': results['params_ga'][7],
        'c': results['params_ga'][8],
        'd': results['params_ga'][9],
        'N': results['params_ga'][10],
        'K': results['params_ga'][11],
        'VELO': results['params_ga'][12],
        'DIFF': results['params_ga'][13],
        'nse_ga': results['nse_ga']
    }, index=[0])
    # Calculate start and end dates for the calibration periods
    if cal_first:
        cal_start_date = cal_start.strftime('%Y-%m-%d')
        cal_end_date = cal_end.strftime('%Y-%m-%d')
    else:
        val_start_date = val_start.strftime('%Y-%m-%d')
        val_end_date = val_end.strftime('%Y-%m-%d')
    results_df.to_csv(results_path.format(g, cal_start_date, cal_end_date), index=False)
    print(f"✅ Done {g} — final NSE = {results['nse_ga']:.4f}")


    # now simulate & plot
    flowsim, routed_direct, routed_base, directflow, baseflow, pet, \
      snow, rain, snowmelt, *_ = run_abcd(
        prcp, tmin, tmax, doy, lat, flowlen,
        snow_par=[results['params_ga'][0], results['params_ga'][1], results['params_ga'][2]],
        pet_par=None,  # still unused
        abcd_par=[results['params_ga'][6], results['params_ga'][7], results['params_ga'][8], results['params_ga'][9]],
        routing_par=[results['params_ga'][10], results['params_ga'][11], results['params_ga'][12], results['params_ga'][13]]
    )
    
    # Convert the simulated flow to a pandas Series
    flowsim = pd.Series(flowsim, index=weather.index, name='ABCD_flow_mm_day')

    plot_validation_figures(obs['flow_mm_day'], flowsim, g, predicted=None, plot_figures=False,
                            cal_first=cal_first, cal_fraction=cal_fraction,
                            path_figures='../results/calibration_results/figures')
    
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--nprocs', type=int, default=1,
                    help='Number of processes to use for parallel processing')
    p.add_argument('--patience', type=int, default=100,
                    help='Early-stop after this many gens w/o improvement.')
    p.add_argument('--min_delta', type=float, default=1e-5,
                   help='Minimum NSE improvement to reset patience.')
    arg = p.parse_args()
    
    nprocs = arg.nprocs
    patience = arg.patience
    min_delta = arg.min_delta

   
    hru_id = pd.read_csv('../data/flow_length.csv', usecols=['hru_id'])
    usgs_gages = [x.zfill(8) for x in hru_id['hru_id'].astype(str).tolist()]
    
    #usgs_gages = usgs_gages[0:12] # for testing, only use the first 10 gages
    
    paths = {
        'livneh': '../data/Livneh_Lusu_extracted_data/',
        'obsflow': '../data/usgs_streamflow/Flow_mm_csv/',
        'shapefiles': '../data/shapefiles/CAMELS/HCDN_nhru_final_671.shp',
        'flowlen': '../data/flow_length.csv',
        'calibration_results': '../results/calibration_results/'
    }
    
    pathlogs = "../results/calibration_results/logs"
    os.makedirs(pathlogs, exist_ok=True)
    
    for fn in os.listdir(pathlogs):
        if fn.startswith("calibration_log_") and fn.endswith(".csv"):
            os.remove(os.path.join(pathlogs, fn))

    # Define the bounds for the parameters
    bounds = [
        (1.0,10.0),(-2,2),(-6,2),
        (0.0,1.0),(0.0,1.0),(0.0,1.0),
        (0.0,1.0),(10,500),(0.01,1.0),(0.01,0.99),
        (1.0,5.0),(0.01,10.0),(0.01,5.0),(0.01,2.0)
    ]

    # Defining the calibration and validation periods
    calibration_first = True # True if the calibration period is the first year of the time series
    calibration_fraction = 0.7 # Fraction of the time series to use for calibration
    
    with Pool(processes=nprocs) as pool:
        pool.map(process_gage,
                 [(g, patience, min_delta, bounds, 
                   calibration_first, calibration_fraction, paths) for g in usgs_gages])
    
    #process_gage((usgs_gages[0],patience,min_delta, bounds, paths))