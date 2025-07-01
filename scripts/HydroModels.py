
# Conda environment: rivercloud

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_func
from numba import njit
import geopandas as gpd
from validation_models import plot_validation_figures



def run_abcd(prcp, tmin, tmax, day_of_year, lat, flowlen,
                          snow_par, pet_par, abcd_par, routing_par):

    
    """
    Run ABCD-Lohmann hydrologic model for a single basin.

    Inputs:
        prcp       : (n_time,) array, precipitation [mm/d]
        tmin, tmax : (n_time,) arrays, min/max temperature [°C]
        day_of_year: (n_time,) array, DOY (1–365)
        lat        : float, basin latitude [degrees]
        flowlen    : float, flow length [m]
        snow_par   : (3,) list or array [m, rain_thres, snow_thres]
        pet_par    : placeholder (unused)
        abcd_par   : (4,) list or array [a, b, c, d]
        routing_par: (4,) list or array [N, K, VELO, DIFF]

    Returns:
        flow, routed_direct, routed_base, directflow, baseflow,
        pet, snow, rain, snowmelt
    """

    # Step 1: Snow model
    snow_par = np.asarray(snow_par, dtype=np.float64)
    rain, snow, snowmelt, snowpack = snow_melt(prcp, tmin, snow_par)
    total_prcp = rain + snowmelt
    #print("snow_melt →", rain.shape, snow.shape, snowmelt.shape, snowpack.shape)

    # Step 2: PET
    pet = safe_pet_hargreaves(tmin, tmax, day_of_year, lat)
    #print("PET →", pet.shape)

    # Step 3: Water balance (ABCD)
    abcd_par = np.asarray(abcd_par, dtype=np.float64)
    directflow, baseflow, uz, lz, evap = abcd(total_prcp, pet, abcd_par)
    #print("ABCD →", directflow.shape, baseflow.shape, uz.shape, lz.shape)

    # Step 4: Unit Hydrographs
    N, K, VELO, DIFF = routing_par
    UH_HRU_direct, UH_HRU_base = generate_HRU_UH([N, K])
    UH_river = generate_channel_UH(flowlen, VELO, DIFF)
    
    #print("  >> UH_HRU_direct:", UH_HRU_direct.shape,
    #  " UH_HRU_base:",   UH_HRU_base.shape,
    #  " UH_river:",      UH_river.shape)

    # Step 5: Routing
    routed_direct, routed_base = routing_lohmann(
        directflow, baseflow, UH_HRU_direct, UH_HRU_base, UH_river
    )
    #print("routed →", routed_direct.shape, routed_base.shape)

    # Step 6: Combine routed flows
    flow = routed_direct + routed_base
    #print("final sim →", flow.shape)
    
    return flow, routed_direct, routed_base, directflow, baseflow, pet, snow, rain, snowmelt, uz, lz, evap



@njit
def snow_melt(prcp, tmin, snow_params, snowpack_initial=0.0):
    """
    Snow accumulation and melt for a single basin.

    Inputs:
        prcp: (n_time,) array, daily precipitation (mm)
        tmin: (n_time,) array, daily minimum temperature (°C)
        snow_params: (3,) tuple or list — [m, rain_thresh, snow_thresh]
        snowpack_initial: float, initial snowpack (default 0)

    Outputs:
        rain: (n_time,) array
        snow: (n_time,) array
        snowmelt: (n_time,) array
        snowpack: (n_time,) array
    """
    m, rain_thresh, snow_thresh = snow_params
    n_time = len(prcp)

    rain = np.zeros(n_time)
    snow = np.zeros(n_time)
    snowmelt = np.zeros(n_time)
    snowpack = np.zeros(n_time)

    snowpack_prev = snowpack_initial

    for t in range(n_time):
        # Determine snow/rain fraction
        if tmin[t] < snow_thresh:
            snow[t] = prcp[t]
            melt_coeff = 0
        elif tmin[t] < rain_thresh:
            frac_snow = (rain_thresh - tmin[t]) / (rain_thresh - snow_thresh)
            snow[t] = prcp[t] * frac_snow
            melt_coeff = m * frac_snow
        else:
            melt_coeff = m

        rain[t] = prcp[t] - snow[t]

        # Snowpack and melt
        total_snow = snowpack_prev + snow[t]
        snowmelt[t] = total_snow * melt_coeff
        snowpack[t] = max(total_snow - snowmelt[t], 0)
        snowpack_prev = snowpack[t]

    return rain, snow, snowmelt, snowpack

@njit
def pet_hargreaves(tmin, tmax, day_of_year, latitude):
    """
    Compute daily PET using Hargreaves method for a single basin.

    Inputs:
        tmin        : (n_days,) array, daily minimum temperatures [°C]
        tmax        : (n_days,) array, daily maximum temperatures [°C]
        day_of_year : (n_days,) array, day of year values (1–365)
        latitude    : float, basin latitude in degrees

    Returns:
        pet         : (n_days,) array of PET values [mm/day]
    """
    GSC = 0.0820  # Solar constant [MJ m⁻² min⁻¹]
    phi = np.radians(latitude)
    day_of_year = np.asarray(day_of_year)
    n_days = len(day_of_year)
    
    pet = np.zeros(n_days)

    for t in range(n_days):
        doy = day_of_year[t]
        dr = 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
        delta = 0.409 * np.sin(2.0 * np.pi * (doy - 81) / 365.0)
        cos_sha = -np.tan(phi) * np.tan(delta)
        if cos_sha < -1.0:
            cos_sha = -1.0
        elif cos_sha > 1.0:
            cos_sha = 1.0
        omega_s = np.arccos(cos_sha)
        et_rad = (24.0 * 60.0 / np.pi) * GSC * dr * (
            omega_s * np.sin(phi) * np.sin(delta) +
            np.cos(phi) * np.cos(delta) * np.sin(omega_s))

        tmean = 0.5 * (tmin[t] + tmax[t])
        td = max(tmax[t] - tmin[t], 0)
        pet[t] = 0.0023 * (tmean + 17.8) * td**0.5 * 0.408 * et_rad

    return pet


def safe_pet_hargreaves(tmin, tmax, day_of_year, latitude):
    tmin = np.asarray(tmin, dtype=np.float64)
    tmax = np.asarray(tmax, dtype=np.float64)
    day_of_year = np.asarray(day_of_year, dtype=np.int64)

    if day_of_year.ndim == 0 or day_of_year.shape == (1,):
        day_of_year = np.full_like(tmin, int(day_of_year[0]) if day_of_year.shape == (1,) else int(day_of_year))

    return pet_hargreaves(tmin, tmax, day_of_year, latitude)


@njit
def abcd(total_prcp, pet, abcd_pars, uz_initial=0.0, lz_initial=0.0):
    """
    ABCD land surface water balance model for a single basin.

    Inputs:
        total_prcp : (n_time,) array, total precipitation [mm/day]
        pet        : (n_time,) array, potential evapotranspiration [mm/day]
        abcd_pars  : (4,) tuple/list of ABCD parameters [a, b, c, d]
        uz_initial : float, initial upper zone storage [mm]
        lz_initial : float, initial lower zone storage [mm]

    Returns:
        direct_runoff : (n_time,) array
        baseflow      : (n_time,) array
        upper_zone    : (n_time,) array
        lower_zone    : (n_time,) array
    """
    n_time = total_prcp.shape[0]
    a, b, c, d = abcd_pars

    # Ensure parameters are in valid ranges
    a = max(a, 1e-6)

    direct_runoff = np.zeros(n_time)
    baseflow = np.zeros(n_time)
    upper_zone = np.zeros(n_time)
    lower_zone = np.zeros(n_time)
    evaporation = np.zeros(n_time)

    uz_prev = uz_initial
    lz_prev = lz_initial

    for t in range(n_time):
        WA = total_prcp[t] + uz_prev

        temp = (WA + b) / (2.0 * a)
        disc = temp * temp - (WA * b) / a
        # Guard against tiny negative rouding errors
        if disc < 0.0:
            disc = 0.0
        EO = temp - np.sqrt(disc)

        # Ensure EO is non-negative. It can't also be larger than WA
        if EO < 0.0:
            EO = 0.0
        elif EO > WA:
            EO = WA

        E = EO * (1.0 - np.exp(-pet[t] / b))

        # Let's be safe and ensure that E is not negative and not large than E0
        if E < 0.0:
            E = 0.0
        elif E > EO:
            E = EO

        Qd = (1.0 - c) * (WA - EO)
        R = c * (WA - EO)

        uz_curr = uz_prev + total_prcp[t] - E - R
        lz_curr = (lz_prev + R) / (1.0 + d)

        direct_runoff[t] = Qd
        baseflow[t] = d * lz_curr
        upper_zone[t] = uz_curr
        lower_zone[t] = lz_curr
        evaporation[t] = E

        uz_prev = uz_curr
        lz_prev = lz_curr

    return direct_runoff, baseflow, upper_zone, lower_zone, evaporation


def generate_HRU_UH(params, KE=12):
    """
    Generate HRU unit hydrographs for a single catchment.

    Parameters:
        params : list or tuple of [N, K] gamma parameters
        KE     : duration of HRU response in days

    Returns:
        UH_direct : (KE,) array
        UH_base   : (KE,) array (delta function at t=0)
    """
    N, K = params
    # hard‐clamp to legal ranges
    K    = max(K,   1e-6)
    
    shape = N
    scale = 1.0 / K

    x_grid = np.linspace(0, 24 * KE, 1000 * KE + 1)
    dx = x_grid[1] - x_grid[0]

    eps=1e-12 # to avoid numerical issues with gamma function when (shape-1) is lower or equal to zero
    gamma_pdf = ( (x_grid+eps) ** (shape - 1)) * np.exp(-x_grid / scale)
    gamma_pdf /= (scale ** shape) * gamma_func(shape)

    # Integrate into daily bins
    edges = np.linspace(0, len(x_grid) - 1, KE + 1, dtype=int)
    UH_direct = np.add.reduceat(gamma_pdf, edges[:-1]) * dx

    UH_base = np.zeros(KE)
    UH_base[0] = 1.0  # delta function for baseflow

    return UH_direct, UH_base


    
def generate_channel_UH(flowlen, velo, diff, UH_DAY=96, DT=3600, LE=2400):
    """
    Generate Green's function-based channel unit hydrograph for a single catchment.

    Parameters:
        flowlen : float, flow length in meters
        velo    : float, wave velocity in m/s
        diff    : float, diffusivity in m²/s
        UH_DAY  : int, duration in days of the routing UH (default 96)
        DT      : int, time step in seconds (default 3600 = 1 hour)
        LE      : int, number of fine time steps to evaluate Green's function (default 2400)

    Returns:
        UH_river : (UH_DAY,) array of daily channel routing unit hydrograph
    """
    # make absolutely sure these are plain scalars
    flowlen = float(flowlen)
    velo    = float(velo)
    diff    = float(diff)

    TMAX = UH_DAY * 24  # hours
    t_grid = DT * np.arange(1, LE + 1)

    pot = ((velo * t_grid - flowlen) ** 2) / (4. * diff * t_grid)
    H = np.where(
        pot <= 69,
        flowlen / (2 * t_grid * np.sqrt(np.pi * t_grid * diff)) * np.exp(-pot),
        0.0,
    )

    H_sum = H.sum()
    UHM = H / H_sum if H_sum > 0 else np.zeros_like(H)
    if H_sum == 0:
        UHM[0] = 1.0

    # Generate finite response (FR) via convolution
    FR = np.zeros(TMAX + 24)
    FR[:24] = 1. / 24.  # Daily pulse

    for t in range(24, TMAX):
        FR[t] = np.dot(FR[t - 24:t][::-1], UHM[:24])

    # Downsample to daily unit hydrograph
    UH_river = np.add.reduceat(FR[:TMAX], np.arange(0, TMAX, 24))

    return UH_river
            


@njit
def routing_lohmann(inflow_direct, inflow_base,
                           UH_HRU_direct, UH_HRU_base, UH_river):
    
    """
    Route flows for a single basin using precomputed unit hydrographs.

    Parameters:
        inflow_direct : (n_time,) array
        inflow_base   : (n_time,) array
        UH_HRU_direct : (n_hru,) array
        UH_HRU_base   : (n_hru,) array
        UH_river      : (n_uh,) array

    Returns:
        directflow    : (n_time,) array
        baseflow      : (n_time,) array
    """
    n_time = inflow_direct.shape[0]
    n_hru = UH_HRU_direct.shape[0]
    n_uh = UH_river.shape[0]
    UH_len = n_hru + n_uh - 1

    # Convolve unit hydrographs (discrete convolution of kernels)
    UH_direct = np.zeros(UH_len)
    UH_base = np.zeros(UH_len)

    for k in range(n_hru):
        for j in range(n_uh):
            UH_direct[k + j] += UH_HRU_direct[k] * UH_river[j]
            UH_base[k + j] += UH_HRU_base[k] * UH_river[j]

    # Normalize the UHs
    sum_direct = np.sum(UH_direct)
    sum_base = np.sum(UH_base)
    if sum_direct > 0:
        UH_direct /= sum_direct
    if sum_base > 0:
        UH_base /= sum_base

    # Route inflows using discrete convolution
    directflow = np.zeros(n_time)
    baseflow = np.zeros(n_time)
    for t in range(n_time):
        for k in range(UH_len):
            if t - k >= 0:
                directflow[t] += inflow_direct[t - k] * UH_direct[k]
                baseflow[t] += inflow_base[t - k] * UH_base[k]

    return directflow, baseflow

            




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    usgs_gages = ['01030500']#'01013500']#, '01022500', ]

    # Read latitude shapefile
    latitudes_df = gpd.read_file('../data/shapefiles/CAMELS/HCDN_nhru_final_671.shp')
    latitudes_df.set_index('hru_id', inplace=True)
    latitudes_df.index = [ x.zfill(8) for x in latitudes_df.index.astype(str) ]

    # Read flow lengths
    flowlen_df = pd.read_csv('../data/flow_length.csv', index_col='hru_id')
    flowlen_df.index = [x.zfill(8) for x in flowlen_df.index.astype(str)]

    for gage in usgs_gages:
        print(f"\nRunning single-basin ABCD for gage {gage}...")

        # Read weather
        weather = pd.read_csv(f'../data/Livneh_Lusu_extracted_data/livneh_lusu_basin_{gage}.csv',
                              index_col='date', parse_dates=True)
        #weather = weather.truncate(before='1950-01-01', after='1950-12-31')
        prcp = weather['prcp_mm'].values
        tmin = weather['tmin_C'].values
        tmax = weather['tmax_C'].values
        day_of_year = weather.index.day_of_year.values

        # Get metadata
        lat = latitudes_df.loc[gage, 'lat_cen']
        flowlen = flowlen_df.loc[gage, 'max_flow_length']

        # Example parameters per basin
        snow_par = [0.1, 1.0, 0.5]
        pet_par = None  # Not used in Hargreaves
        abcd_par = [0.9, 400.0, 0.21, 0.25]
        routing_par = [1.0, 0.5, 0.1, 0.01]  # N, K, VELO, DIFF

        # Call single-basin simulation
        flow, routed_direct, routed_base, directflow, baseflow, pet, snow, rain, snowmelt, upperzone, lowerzone, evap = \
            run_abcd(prcp, tmin, tmax, day_of_year, lat, flowlen,
                                  snow_par, pet_par, abcd_par, routing_par)

        # Convert flow to pandas DataFrame
        simflow = pd.DataFrame(flow, index=weather.index, columns=['ABCD_flow_mm_day'])
        # Read observed flow
        obs_path = f'../data/usgs_streamflow/Flow_mm_csv/{gage}_observed_flow.csv'
        obsflow = pd.read_csv(obs_path, index_col='date', usecols=['date','flow_mm_day'], parse_dates=True)

        plot_validation_figures(obsflow, simflow, gage, predicted=None, plot_figures=False,
                            cal_first=True, cal_fraction=0.7,
                            path_figures='../results/calibration_results/figures')
        
        print(f"{gage} | Final flow sample: {flow[-5:]}")
        