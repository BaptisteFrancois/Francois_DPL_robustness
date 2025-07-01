
import pandas as pd
import numpy as np
from tqdm import tqdm
# Read the basin_physical_characteristics.txt file
# Note: the separator is whitespace, so we use sep='\s+'
basin_physical_characteristics = pd.read_csv(
    '../data/basin_physical_characteristics.txt', sep='\s+', header=0)


for _, basin in tqdm(basin_physical_characteristics.iterrows(), 
                     desc='Processing basins', total=len(basin_physical_characteristics)):
    basin_huc = str(int(basin['BASIN_HUC'])).zfill(2)
    basin_id = str(int(basin['BASIN_ID'])).zfill(8)  # Ensure the ID is zero-padded to 8 digits
    basin_size = basin['Size(km2)']

    # Read the observed flow data for the basin
    obs_flow = pd.read_csv(
        f'../data/usgs_streamflow/{basin_huc}/{basin_id}_streamflow_qc.txt', sep='\s+', header=None)
    
    # ID, year, momth, day, flow_cms, flag
    obs_flow.columns = ['ID', 'year', 'month', 'day', 'flow_cms', 'flag']
    # Using year, month, day to create a timestamp and use it as index
    obs_flow['date'] = pd.to_datetime(obs_flow[['year', 'month', 'day']])
    obs_flow.set_index('date', inplace=True)
    obs_flow.drop(columns=['ID', 'year', 'month', 'day', 'flag'], inplace=True)

    # Replace negative flow values with NaN
    obs_flow['flow_cms'] = obs_flow['flow_cms'].replace(-999.0, np.nan)

    # Convert flow from cms to mm/day
    obs_flow['flow_mm_day'] = obs_flow['flow_cms'] * 86400 / (basin_size *1e6) * 1000  # Convert cms to mm/day
    obs_flow['flow_mm_day'] = obs_flow['flow_mm_day'].round(2)  # Round to 3 decimal places

    # Save the observed flow data to a CSV file
    obs_flow.to_csv(
        f'../data/usgs_streamflow/Flow_mm_csv/{basin_id}_observed_flow.csv', header=True)

