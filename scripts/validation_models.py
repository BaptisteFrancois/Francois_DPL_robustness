
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



def plot_validation_figures(obs_ts, hydromodel, name, 
                            predicted=None, plot_figures=True,
                            cal_first=True, cal_fraction=0.7,
                            path_figures='../results/calibration_results/figures/'):

    """
    Plot validation figures for the hydrological model.
    Parameters
    ----------
    obs_ts : pd.DataFrame or pd.Series
        Observed time series of flow.
    hydromodel : pd.DataFrame or pd.Series
        Simulated time series of flow from the hydrological model.
    name : str
        Name of the hydrological model.
    predicted : pd.DataFrame or pd.Series, optional
        Predicted time series of flow (if available).
    plot_figures : bool, optional
        Whether to plot the figures (default is True).
    cal_first : bool, optional
        Whether to use the first part of the time series for calibration (default is True).
    cal_fraction : float, optional
        Fraction of the time series to use for calibration (default is 0.7).
    tsdate_start : pd.Timestamp, optional
        Start date of the time series (if available).
    path_figures : str, optional
        Path to save the figures (default is '../results/calibration_results/figures/').
    Returns
    -------
    None
    """

    # NSE and bias functions
    nse_lamb= lambda obs, sim: 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)
    # Relative bias in percentage   
    bias_relative_lamb= lambda obs, sim: np.mean(sim - obs) / obs.mean() * 100  


    if not os.path.exists(path_figures):
        os.makedirs(path_figures)


    # Align the time series data to ensure they have the same index
    obs_ts, hydromodel = obs_ts.align(hydromodel, join='inner', axis=0)
    if isinstance(predicted, pd.Series):
        obs_ts, predicted = obs_ts.align(predicted, join='inner', axis=0)

    

    # Split the time series into calibration and validation periods
    if cal_fraction < 1:
        if cal_first:  # the beginning of the time series is the calibration period
            cal_len = int(len(obs_ts) * cal_fraction)
            cal_start = obs_ts.index[0]
            cal_end = obs_ts.index[cal_len - 1]
            val_start = obs_ts.index[cal_len]
            val_end = obs_ts.index[-1]
        else:  # the end of the time series is the calibration period
            val_len = int(len(obs_ts) * cal_fraction)
            val_start = obs_ts.index[-val_len]
            val_end = obs_ts.index[-1]
            cal_start = obs_ts.index[0]
            cal_end = obs_ts.index[-val_len - 1]
    else:  # no validation period, only calibration
        cal_start = obs_ts.index[0]
        cal_end = obs_ts.index[-1]


    # Select the calibration and validation periods
    obs_cal = obs_ts[(obs_ts.index >= cal_start) & (obs_ts.index <= cal_end)]
    hydromodel_cal = hydromodel[(hydromodel.index >= cal_start) & (hydromodel.index <= cal_end)]
    if isinstance(predicted, pd.Series):
        predicted_cal = predicted[(predicted.index >= cal_start) & (predicted.index <= cal_end)]
    if cal_fraction < 1:
        obs_val = obs_ts[(obs_ts.index >= val_start) & (obs_ts.index <= val_end)]
        hydromodel_val = hydromodel[(hydromodel.index >= val_start) & (hydromodel.index <= val_end)]
        if isinstance(predicted, pd.Series):
            predicted_val = predicted[(predicted.index >= val_start) & (predicted.index <= val_end)]
    
    
    # Resample the time series to monthly and annual water year (WY) sums for calibration 
    obs_monthly = obs_ts.resample('ME').mean()
    obs_monthly_cal = obs_cal.resample('ME').mean()
    hydromodel_monthly_cal = hydromodel_cal.resample('ME').mean()
    obs_wy = obs_ts.resample('YE-SEP').mean()
    obs_wy = obs_wy[1:-1]  # Drop first and last year to avoid incomplete data
    obs_wy_cal = obs_cal.resample('YE-SEP').mean()
    obs_wy_cal = obs_wy_cal[1:-1]  # Drop first and last year to avoid incomplete data
    hydromodel_wy_cal = hydromodel_cal.resample('YE-SEP').mean()
    hydromodel_wy_cal = hydromodel_wy_cal[1:-1]  # Drop first and last year to avoid incomplete data

    if isinstance(predicted, pd.Series):
        predicted_monthly_cal = predicted_cal.resample('ME').mean()
        predicted_wy_cal = predicted_cal.resample('YE-SEP').mean()
        predicted_wy_cal = predicted_wy_cal[1:-1]  # Drop first and last year to avoid incomplete data

    # Resample the time series to monthly and annual water year (WY) sums for validation (if needed)
    if cal_fraction < 1:
        obs_monthly_val = obs_val.resample('ME').mean()
        hydromodel_monthly_val = hydromodel_val.resample('ME').mean()
        obs_wy_val = obs_val.resample('YE-SEP').mean()
        obs_wy_val = obs_wy_val[1:-1]  # Drop first and last year to avoid incomplete data
        hydromodel_wy_val = hydromodel_val.resample('YE-SEP').mean()
        hydromodel_wy_val = hydromodel_wy_val[1:-1]  # Drop first and last year to avoid incomplete data
        if isinstance(predicted, pd.Series):
            predicted_monthly_val = predicted_val.resample('ME').mean()
            predicted_wy_val = predicted_val.resample('YE-SEP').mean()
            predicted_wy_val = predicted_wy_val[1:-1]  # Drop first and last year to avoid incomplete data


    # Plot a figure with 4 subplots.
    # The figure has 2 rows and 2 columns. The first row has the monthly flow time series.
    # The second row has threee subplots:
    # - The annual time series of the observed, simulated and downscaled flow.
    # - The average day of year (DOY) cycle of the observed, simulated and downscaled flow.
    # - The ECDF of the daily flows

    try:
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(nrows=2, ncols=3, figure=fig)

        # Monthly flow time series

        # nse and bias (remove NaN values)
        mask = ~np.isnan(obs_monthly_cal.values) & ~np.isnan(hydromodel_monthly_cal.values)
        nse_hydromodel_month_cal = round(nse_lamb(obs_monthly_cal.values[mask], hydromodel_monthly_cal.values[mask]),2)
        bias_hydromodel_month_cal = round(bias_relative_lamb(obs_monthly_cal.values[mask], hydromodel_monthly_cal.values[mask]), 2)
        if cal_fraction < 1:
            mask_val = ~np.isnan(obs_monthly_val.values) & ~np.isnan(hydromodel_monthly_val.values)
            nse_hydromodel_month_val = round(nse_lamb(obs_monthly_val.values[mask_val], hydromodel_monthly_val.values[mask_val]),2)
            bias_hydromodel_month_val = round(bias_relative_lamb(obs_monthly_val.values[mask_val], hydromodel_monthly_val.values[mask_val]), 2)

        ax1 = fig.add_subplot(gs[0, :3])  # First row, all columns
        ax1.plot(obs_monthly, label='Observed', color='black')
        ax1.plot(hydromodel_monthly_cal,
                label='ABCD-cal (NSE={}; Bias(%)={})'.format(
                    nse_hydromodel_month_cal, bias_hydromodel_month_cal), 
                color='#7fcdbb')
        if cal_fraction < 1:
            ax1.plot(hydromodel_monthly_val,
                    label='ABCD-val (NSE={}; Bias(%)={})'.format(
                    nse_hydromodel_month_val, bias_hydromodel_month_val), 
                    color='#2c7fb8')

        if isinstance(predicted, pd.Series):
            mask = ~np.isnan(obs_monthly_cal.values) & ~np.isnan(predicted_monthly_cal.values)
            nse_predicted_monthly_cal = \
                round(nse_lamb(obs_monthly_cal.values[mask], predicted_monthly_cal.values[mask]), 2)
            bias_predicted_monthly_cal = \
                round(bias_relative_lamb(obs_monthly_cal.values[mask], predicted_monthly_cal.values[mask]), 2)

            ax1.plot(predicted_monthly_cal, 
                    label='ML-cal (NSE={}; Bias(%)={})'.format(
                        nse_predicted_monthly_cal, bias_predicted_monthly_cal), 
                    color='#feb24c')
            if cal_fraction < 1:
                mask = ~np.isnan(obs_monthly_val.values) & ~np.isnan(predicted_monthly_val.values)
                nse_predicted_monthly_val = \
                    round(nse_lamb(obs_monthly_val.values[mask], predicted_monthly_val.values[mask]), 2)
                bias_predicted_monthly_val = \
                    round(bias_relative_lamb(obs_monthly_val.values[mask], predicted_monthly_val.values[mask]), 2)
                ax1.plot(predicted_monthly_val, 
                        label='ML-val (NSE={}; Bias(%)={})'.format(
                        nse_predicted_monthly_val, bias_predicted_monthly_val), 
                        color='#f03b20')

        ax1.set_ylabel('Flow (mm)')
        ax1.set_title('Average Monthly Flow Time Series')
        ax1.legend()
        ax1.grid(True)

        # Annual flow time series
        ax2 = fig.add_subplot(gs[1, 0])  # Second row, first column
        ax2.plot(obs_wy, label='Observed', color='black')
        mask = ~np.isnan(obs_wy_cal.values) & ~np.isnan(hydromodel_wy_cal.values)
        nse_hydromodel_wy_cal = round(nse_lamb(obs_wy_cal.values[mask], hydromodel_wy_cal.values[mask]), 2)
        bias_hydromodel_wy_cal = round(bias_relative_lamb(obs_wy_cal.values[mask], hydromodel_wy_cal.values[mask]), 2)
        ax2.plot(hydromodel_wy_cal, 
                label='ABCD-cal (NSE={}; Bias(%)={})'.format(
                    nse_hydromodel_wy_cal, bias_hydromodel_wy_cal), 
                color='#7fcdbb')
        if cal_fraction < 1:
            mask = ~np.isnan(obs_wy_val.values) & ~np.isnan(hydromodel_wy_val.values)
            nse_hydromodel_wy_val = \
                round(nse_lamb(obs_wy_val.values[mask], hydromodel_wy_val.values[mask]), 2)
            bias_hydromodel_wy_val = \
                round(bias_relative_lamb(obs_wy_val.values[mask], hydromodel_wy_val.values[mask]), 2)
            ax2.plot(hydromodel_wy_val, 
                    label='ABCD-val (NSE={}; Bias(%)={})'.format(
                        nse_hydromodel_wy_val, bias_hydromodel_wy_val), 
                    color='#2c7fb8')


        if isinstance(predicted, pd.Series):
            mask = ~np.isnan(obs_wy_cal.values) & ~np.isnan(predicted_wy_cal.values)
            nse_predicted_wy_cal = round(nse_lamb(obs_wy_cal.values[mask], predicted_wy_cal.values[mask]), 2)
            bias_predicted_wy_cal = round(bias_relative_lamb(obs_wy_cal.values[mask], predicted_wy_cal.values[mask]), 2)

            ax2.plot(predicted_wy_cal, 
                    label='ML-cal (NSE={}; Bias(%)={})'.format(
                        nse_predicted_wy_cal, bias_predicted_wy_cal), 
                    color='#feb24c')
            if cal_fraction < 1:
                mask = ~np.isnan(obs_wy_val.values) & ~np.isnan(predicted_wy_val.values)
                nse_predicted_wy_val = round(nse_lamb(obs_wy_val.values[mask], predicted_wy_val.values[mask]), 2)
                bias_predicted_wy_val = round(bias_relative_lamb(obs_wy_val.values[mask], predicted_wy_val.values[mask]), 2)
                ax2.plot(predicted_wy_val, 
                        label='ML-val (NSE={}; Bias(%)={})'.format(
                        nse_predicted_wy_val, bias_predicted_wy_val), 
                        color='#f03b20')
                
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Flow (mm)')
        ax2.set_title('Average Annual Flow Time Series')
        ax2.legend()

        # Average day of year (DOY) cycle
        ax3 = fig.add_subplot(gs[1, 1])  # Second row, second column
        # Calculate the average DOY cycle for observed, simulated and downscaled flow
        obs_doy_cycle = obs_ts.groupby(obs_ts.index.dayofyear).mean()
        sim_doy_cycle = hydromodel.groupby(hydromodel.index.dayofyear).mean()
        ax3.plot(obs_doy_cycle, label='Observed', color='black')
        ax3.plot(sim_doy_cycle, label='ABCD', color='#1d91c0')

        if isinstance(predicted, pd.Series):
            predicted_doy_cycle = predicted.groupby(predicted.index.dayofyear).mean()
            ax3.plot(predicted_doy_cycle, label='Predicted', color='#fd8d3c')
        
        ax3.set_xlabel('Day of Year')
        ax3.set_ylabel('Flow (mm)')
        ax3.grid()
        ax3.set_title('Average Day of Year Cycle')
        ax3.legend()

        # Empirical CDF of daily flows
        ax4 = fig.add_subplot(gs[1, 2])  # Second row, third column
        # Calculate the empirical CDF for observed, simulated and downscaled flow
        obs_ts_series = obs_ts[obs_ts.notna()]['flow_mm_day'] if isinstance(obs_ts, pd.DataFrame) else obs_ts.dropna()
        obs_ecdf = sm.distributions.ECDF(obs_ts_series.to_numpy())
        hydromodel_series = hydromodel[hydromodel.notna()]['ABCD_flow_mm_day'] if isinstance(hydromodel, pd.DataFrame) else hydromodel.dropna()
        obs_ts_series, hydromodel_series = obs_ts_series.align(hydromodel_series, join='inner', axis=0)
        sim_ecdf = sm.distributions.ECDF(hydromodel_series.to_numpy())
        x_obs = np.sort(obs_ts_series.dropna())
        x_sim = np.sort(hydromodel_series.dropna())
        if isinstance(predicted, pd.Series):
            predicted_ecdf = sm.distributions.ECDF(predicted.dropna())
            x_predicted = np.sort(predicted.dropna())
        ax4.plot(x_obs, obs_ecdf(x_obs), label='Observed', color='black')
        
        mask = ~np.isnan(obs_ts_series.values) & ~np.isnan(hydromodel_series.values)
        nse_hydromodel_daily = round(nse_lamb(obs_ts_series.values[mask], hydromodel_series.values[mask]), 2)
        ax4.plot(x_sim, sim_ecdf(x_sim), label='ABCD (NSE daily={})'.format(
            nse_hydromodel_daily), color='#1d91c0')
        if isinstance(predicted, pd.Series):
            mask = ~np.isnan(obs_ts_series.values) & ~np.isnan(predicted.values)
            nse_predicted_daily = round(nse_lamb(obs_ts_series.values[mask], predicted.values[mask]), 2)
            ax4.plot(x_predicted, predicted_ecdf(x_predicted), label='Predicted (NSE daily={})'.format(
                nse_predicted_daily), color='#fd8d3c')
        ax4.set_xlabel('Flow (mm)')  
        ax4.set_ylabel('Empirical CDF')
        ax4.set_title('Empirical CDF of Daily Flows')
        ax4.grid()
        ax4.legend()
        # Log scale the x-axis for better visibility
        ax4.set_xscale('log')

        plt.suptitle(f'Flow Evaluation for {name}', fontsize=16)

        plt.tight_layout()
        if isinstance(predicted, pd.Series):
            plt.savefig(f'{path_figures}/{name}_Inflow_analysis_w_predicted.png', dpi=150)
        else:   
            plt.savefig(f'{path_figures}/{name}_Inflow_analysis.png', dpi=150)
                    
        if plot_figures:
            plt.show()
        plt.close()
    except Exception as e:
        import traceback
        print(f"Error plotting validation figures for {name}: {e}")
        traceback.print_exc()
        return