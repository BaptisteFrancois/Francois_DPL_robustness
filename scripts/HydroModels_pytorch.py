

import torch
import torch.nn.functional as F
from utils import (    
    snow_melt_torch,
    pet_hargreaves_torch,
    abcd_torch,
    generate_HRU_UH_torch,
    generate_channel_UH_torch,
    routing_lohman_torch
)
from typing import Tuple, Union, List, Dict, Optional


# Core simulator class for PyTorch hydro models. Expects 'physical' parameters, not NN outputs.
class HydroModel_pytorch(torch.nn.Module):
    
    """
    Base class for PyTorch hydro models.
    This class can be extended to implement specific hydro models.
    """


    def __init__(self,            
                 #  Geography and basin parameters
                 flowlen:  Union[float, torch.Tensor],  # flow length in meters
                 latitude: Union[float, torch.Tensor],  # latitude in degrees
                 # initial-state defaults
                 snowpack_initial: Union[float, torch.Tensor] = 0.0,
                 uz_initial: Union[float, torch.Tensor] = 0.0,
                 lz_initial: Union[float, torch.Tensor] = 0.0,
                 # UH generation settings
                 UH_DAY: int = 96,
                 KE:     int = 12,
                 DT:     int = 3600,
                 LE:     int = 2400):
        super().__init__()
        # 1D-UH settings
        self.UH_DAY = UH_DAY
        self.KE = KE
        self.DT = DT
        self.LE = LE

        # register flowlen & latitude as buffers (not trained)
        self.register_buffer('flowlen', torch.as_tensor(flowlen, dtype=torch.float32))
        self.register_buffer('latitude', torch.as_tensor(latitude, dtype=torch.float32))

        # Initial conditions (left as floats, can be tensors later)
        self.snowpack_initial = snowpack_initial  # initial snowpack [mm]
        self.uz_initial = uz_initial  # initial upper zone [mm]
        self.lz_initial = lz_initial  # initial lower zone [mm]

    def forward(self,
                # climate forcings
                prcp: torch.Tensor,
                tmin: torch.Tensor,
                tmax: torch.Tensor,
                day_of_year: torch.Tensor,
                # model parameters
                snow_params:    Tuple[Union[float, torch.Tensor],
                                      Union[float, torch.Tensor],
                                      Union[float, torch.Tensor]],
                abcd_params:    Tuple[Union[float, torch.Tensor],
                                      Union[float, torch.Tensor],
                                      Union[float, torch.Tensor],
                                      Union[float, torch.Tensor]],
                hru_params:     Tuple[Union[float, torch.Tensor],
                                      Union[float, torch.Tensor]],
                channel_params: Tuple[Union[float, torch.Tensor],
                                      Union[float, torch.Tensor]]
               ) -> Dict[str, torch.Tensor]:

        # Unpack parameters
        m, rain_thr, snow_thr = snow_params
        a, b, c, d = abcd_params
        shape, rate = hru_params
        velo, diff = channel_params

        # snow melt routine
        rain, snow, snowmelt, snowpack = snow_melt_torch(
            prcp, tmin, (m, rain_thr, snow_thr), snowpack_initial=self.snowpack_initial
        )

        # PET
        pet = pet_hargreaves_torch(tmin, tmax, day_of_year, self.latitude)

        # ABCD water balance
        Qd, Qb, uz, lz, E = abcd_torch(
            rain + snowmelt, pet, (a, b, c, d), self.uz_initial, self.lz_initial
        )

        # HRU UHs
        UH_HRU_direct, UH_HRU_base = generate_HRU_UH_torch(
            (shape, rate), KE=self.KE
        )

        # Channel UH
        UH_river = generate_channel_UH_torch(
            self.flowlen, velo, diff,
            UH_DAY=self.UH_DAY, DT=self.DT, LE=self.LE, device=prcp.device
        )

        # Route flows using Lohman's method
        directflow, baseflow = routing_lohman_torch(
            Qd, Qb, UH_HRU_direct, UH_HRU_base, UH_river
        )


        # return outputs
        return {
            'rain': rain,
            'snow': snow,
            'snowmelt': snowmelt,
            'snowpack': snowpack,
            'pet': pet,
            'Qd': Qd,
            'Qb': Qb,
            'uz': uz,
            'lz': lz,
            'E': E,
            'directflow': directflow,
            'baseflow': baseflow
        }
    



# ParamNet predicts HydroModel_pytorch parameters
class ParamNet(torch.nn.Module):

    """ParamNet predicts parameters for the HydroModel_pytorch.
    """

    def __init__(
            self,
            in_features: int,
            bounds: List[Tuple[float, float]], # one (min, max) tuple for each parameter
            hidden: int = 32
    ):
        
        """
        ParamNet predicts parameters for the HydroModel.
        Args:
            in_features (int): Number of input features.
            hidden (int): Number of hidden units in the first layer.
        """
        super(ParamNet, self).__init__()
        self.bounds = bounds
        out_dim = len(bounds)    # number of parameters to predict
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor):
        # Pass through the network
        z = self.net(x)   # (batch, out_dim)

        # build tensors of lower and upper bounds
        lows = x.new_tensor([b[0] for b in self.bounds])  # (out_dim,)
        highs = x.new_tensor([b[1] for b in self.bounds])  # (out_dim,)

        # map z -> [0,1] via sigmoid, then scale -> [lows, highs] 
        p = lows + (highs - lows) * torch.sigmoid(z)  # (batch, out_dim)

        # split into individual parameters
        snow_z, abcd_z, hru_z, chn_z = torch.split(p, [3, 4, 2, 2], dim=-1)

        return snow_z, abcd_z, hru_z, chn_z     # each (batch, *)


# Wrapper that uses either NN-predicted or user-provided parameters
class RunHydroModel(torch.nn.Module):
    
    """Wrapper for HydroModel_pytorch that can use either a ParamNet to predict parameters
    or accept parameters directly."""
    
    def __init__(self,
                 # Geography and basin parameters
                 flowlen:  Union[float, torch.Tensor],  # flow length in meters
                 latitude: Union[float, torch.Tensor],  # latitude in degrees
                 # (if used) Number of input features for the ParamNet)
                 feature_dim: int = None,
                 # bounds for the parameters
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 # initial-state defaults
                 snowpack_initial: Union[float, torch.Tensor] = 0.0,
                 uz_initial: Union[float, torch.Tensor] = 0.0,
                 lz_initial: Union[float, torch.Tensor] = 0.0,
                 # UH generation settings
                 UH_DAY: int = 96,
                 KE: int = 12,
                 DT: int = 3600,
                 LE: int = 2400
    ):
        
        """        RunHydroModel initializes the hydro model with either a ParamNet or fixed parameters.
        Args:
            flowlen (float or torch.Tensor): Flow length in meters.
            latitude (float or torch.Tensor): Latitude in degrees.
            feature_dim (int, optional): Number of input features for the ParamNet. If None, uses fixed parameters.
            bounds (List[Tuple[float, float]], optional): Bounds for the parameters. If None, uses default bounds.
            snowpack_initial (float or torch.Tensor): Initial snowpack in mm.   
            uz_initial (float or torch.Tensor): Initial upper zone in mm.
            lz_initial (float or torch.Tensor): Initial lower zone in mm.
        """
        super().__init__()

        # Assign bounds if not provided
        if bounds is None:
            bounds = (
                    (1.0,10.0),(-2,2),(-6,2),
                    (0.0,1.0),(10,500),(0.01,1.0),(0.01,0.99),
                    (1.0,5.0),(0.01,10.0),(0.01,5.0),(0.01,2.0)
            )
        self.bounds = bounds

        # Choose static vs. learnable mode
        self.use_nn = feature_dim is not None
        if self.use_nn:
            self.param_net = ParamNet(feature_dim, bounds)
        else:
            self.param_net = None

        # core simulator class
        self.hydro_model = HydroModel_pytorch(
            flowlen=flowlen,
            latitude=latitude,
            snowpack_initial=snowpack_initial,
            uz_initial=uz_initial,
            lz_initial=lz_initial,
            UH_DAY=UH_DAY,
            KE=KE,
            DT=DT,
            LE=LE
        )

    def forward(self,
                # climate forcings
                prcp: torch.Tensor,
                tmin: torch.Tensor,
                tmax: torch.Tensor,
                day_of_year: torch.Tensor,
                *,
                # input features for ParamNet (if used)
                features: torch.Tensor = None,
                # model parameters (if not using ParamNet)
                snow_params: Tuple[Union[float, torch.Tensor],
                                    Union[float, torch.Tensor],
                                    Union[float, torch.Tensor]] = None,
                abcd_params: Tuple[Union[float, torch.Tensor],
                                    Union[float, torch.Tensor],
                                    Union[float, torch.Tensor],
                                    Union[float, torch.Tensor]] = None,
                hru_params: Tuple[Union[float, torch.Tensor],
                                    Union[float, torch.Tensor]] = None,
                channel_params: Tuple[Union[float, torch.Tensor],
                                        Union[float, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        
        # If using ParamNet, predict parameters
        if self.use_nn:
            if features is None:
                raise ValueError("Features must be provided when using ParamNet.")
            snow_params, abcd_params, hru_params, channel_params = \
                self.param_net(features)

        # If not using ParamNet, ensure parameters are provided
        else:
            if (snow_params is None or abcd_params is None or
                    hru_params is None or channel_params is None):
                raise ValueError("Parameters must be provided when not using ParamNet.")

        # Run the hydro model with the provided parameters
        return self.hydro_model(
            prcp, tmin, tmax, day_of_year,
            snow_params=snow_params,
            abcd_params=abcd_params,
            hru_params=hru_params,
            channel_params=channel_params
        )
            

if __name__ == "__main__":

    import os, glob
    import pandas as pd
    import geopandas as gpd
    from validation_models import plot_validation_figures
    from HydroModels import run_abcd
    

    # Run the HydroModel_pytorch with example parameters

    # Example input tensors
    usgs_gages = ['01030500', '01013500', '01022500', ]

    # Read latitude shapefile
    latitudes_df = gpd.read_file('../data/shapefiles/CAMELS/HCDN_nhru_final_671.shp')
    latitudes_df.set_index('hru_id', inplace=True)
    latitudes_df.index = [ x.zfill(8) for x in latitudes_df.index.astype(str) ]

    # Read flow lengths
    flowlen_df = pd.read_csv('../data/flow_length.csv', index_col='hru_id')
    flowlen_df.index = [x.zfill(8) for x in flowlen_df.index.astype(str)]

    for gage in usgs_gages:
        print(f"\nRunning single-basin ABCD for gage {gage}...")

        # Get the latitude and flow length for the gage
        latitude = latitudes_df.loc[gage.zfill(8), 'lat_cen']
        flowlen = flowlen_df.loc[gage.zfill(8), 'max_flow_length']

        # Create the model
        model = HydroModel_pytorch(
            flowlen=flowlen,
            latitude=latitude,
            snowpack_initial=0.0,
            uz_initial=0.0,
            lz_initial=0.0,
            UH_DAY=96,
            KE=12,
            DT=3600,
            LE=2400
        )

        # Climate data
        # Read weather
        weather = pd.read_csv(f'../data/Livneh_Lusu_extracted_data/livneh_lusu_basin_{gage}.csv',
                              index_col='date', parse_dates=True)
        #weather = weather.truncate(before='1950-01-01', after='1950-12-31')
        prcp = weather['prcp_mm'].values
        tmin = weather['tmin_C'].values
        tmax = weather['tmax_C'].values
        day_of_year = weather.index.day_of_year.values


        # Read model parameters (calibrated using the non-PyTorch version)
        param_file = glob.glob(f'../results/calibration_results/abcd_parameters_{gage}_*.csv')[0]
        if not param_file:
            raise FileNotFoundError(f"No parameter file found for gage {gage}.")
        # Read the parameters
        param = pd.read_csv(param_file)
        # Extract parameters
        snow_params = (param['m'].values[0], param['rain_thr'].values[0], param['snow_thr'].values[0])
        abcd_params = (param['a'].values[0], param['b'].values[0], 
                       param['c'].values[0], param['d'].values[0])
        hru_params = (param['N'].values[0], param['K'].values[0])
        channel_params = (param['VELO'].values[0], param['DIFF'].values[0])


        # Run the non-PyTorch version
        pet_params = (0.0, 0.0, 0.0, 0.0)
        routing_params = (hru_params[0], hru_params[1], channel_params[0], channel_params[1])
        abcd_flow, routed_direct, routed_base, Qd, Qb, pet, snow, rain, snowmelt, snowpack, upperzone, lowerzone, evap \
             = run_abcd(prcp, tmin, tmax, day_of_year, latitude, flowlen,
                             snow_params, pet_params, abcd_params, routing_params)
        simflow_non_pytorch = pd.DataFrame(abcd_flow, index=weather.index, columns=['ABCD_flow_mm_day'])



        # Convert to tensors
        prcp = torch.tensor(prcp, dtype=torch.float32)
        tmin = torch.tensor(tmin, dtype=torch.float32)
        tmax = torch.tensor(tmax, dtype=torch.float32)
        day_of_year = torch.tensor(day_of_year, dtype=torch.int64)

        

        # Run the model
        outputs = model(
            prcp, tmin, tmax, day_of_year,
            snow_params=snow_params,
            abcd_params=abcd_params,
            hru_params=hru_params,
            channel_params=channel_params
        )

        # Read the observed streamflow data
        obs_path = f'../data/usgs_streamflow/Flow_mm_csv/{gage}_observed_flow.csv'
        obsflow = pd.read_csv(obs_path, index_col='date', usecols=['date','flow_mm_day'], parse_dates=True)

                
        # Convert PyTorch outputs to numpy for comparison
        directflow = outputs['directflow'].cpu().numpy()
        baseflow = outputs['baseflow'].cpu().numpy()
        abcd_pytorch_flow = directflow + baseflow
        simflow_pytorch = pd.DataFrame(abcd_pytorch_flow, index=weather.index, columns=['ML_flow_mm_day'])

        plot_validation_figures(obsflow, simflow_non_pytorch, gage, predicted=simflow_pytorch, plot_figures=True,
                            cal_first=True, cal_fraction=0.7,
                            path_figures='../results/calibration_results/figures_test/')

        

