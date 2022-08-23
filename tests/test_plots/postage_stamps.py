import cartopy.crs as ccrs
import hvplot.xarray
import hvplot.pandas
import holoviews as hv
from holoviews import opts
from matplotlib.figure import Figure
import xarray as xr
xr.set_options(keep_attrs=True)
import panel as pn
import numpy as np

def postage_stamps(ds, var):
    
    '''
    outputs postage stamp type figure
    showing anomaly maps for a given variable.
    
    currently, the warming threshold is hardcoded.
    
    ds = xarray dataset
    var = variable (string), currently tuned for T at 2m 
    (this is because the colorbar and cmap are hardcoded).
    '''

    var = data[var_str]
    var.attrs['reference_range'] = '1981','2010'# hist threshold

    # year ranges for each warming threshold for each model
    warming_15 = [(str(y-15),str(y+14)) for y in [2044,2050,2045,2053,2053]]
    warming_2 = [(str(y-15),str(y+14)) for y in [2055,2061,2056,2072,2069]]
    warming_3 = [(str(y-15),str(y+14)) for y in [2078,2077,2073,np.nan,np.nan]]
    warming_4 = [(str(y-15),str(y+14)) for y in [np.nan,np.nan,2091,np.nan,np.nan]]

    ### generate weights based off days per month
    month_length = data.time.dt.days_in_month
    ### first monthly weights, ie, days in month / days in year
    mon_wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    ### then annual weights, days in year / total days
    # wgts.groupby("time.year").sum(xr.ALL_DIMS) # check that annual weights make sense

    ### want to ensure nans do not impact the weighting
    # code from https://ncar.github.io/esds/posts/2021/yearly-averages-xarray/

    # Setup our masking for nan values
    cond = var.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    var_sum = (var * mon_wgts).resample(time="AS").sum(dim="time")
    # Calculate the denominator
    ones_out = (ones * mon_wgts).resample(time="AS").sum(dim="time")
    # weighted mean
    wgt_ann_mean = var_sum / ones_out

    var_hist = var.sel(time=slice(*wgt_ann_mean.reference_range))

    ### now for annual weights, days in year / total days
    ### need to do this for each time period
    ### note: each time period differs by the year the  
    ### warming threshold is reached.
    year_length = month_length.groupby("time.year").sum() 
    year_length.name = "days_in_year"
    year_length = year_length.assign_coords(year=wgt_ann_mean.time.values)

    hist_time_slice = year_length.sel(year=slice(*var.reference_range))
    hist_ann_wgts = hist_time_slice / hist_time_slice.sum()
    hist_ann_wgts = hist_ann_wgts.rename({'year' : 'time'})

    # # get the weighted 30-year historical statistics
    ann_hist_mean = wgt_ann_mean.sel(time=slice(*wgt_ann_mean.reference_range))
    hist_wgtd = ann_hist_mean.weighted(hist_ann_wgts)
    hist_mean = hist_wgtd.mean("time")
    hist_mean.name = 'Mean'

    # other historical stats
    var_hist_median = var_hist.median("time")
    var_hist_median.name = 'Median'
    var_hist_max = var_hist.max("time")
    var_hist_max.name = 'Max'
    var_hist_min = var_hist.min("time")
    var_hist_min.name = 'Min'

    # historical dataset of relevant statistics
    var_hist_stat = xr.merge([hist_mean,var_hist_median,var_hist_max,var_hist_min])

    # intialize the plot
    fig = Figure(figsize=(8, 10))
    axs = fig.subplots(nrows=5,ncols=4, 
                           subplot_kw={'projection' : ccrs.LambertConformal()}, 
                           sharey=True,sharex=True)

    my_statistics = [v for v in var_hist_stat.data_vars]
    my_simulations = ['cesm2', 'cnrm-esm2-1', 'ec-earth3-veg', 'fgoals-g3', 'mpi-esm1-2-lr']

    for i,sim in enumerate(var_hist_stat['simulation']):

        #### compute the anomalies
        # need the same stats as above
        # but for the given warming scenario
        var_warm = var.sel(time=slice(*warming_2[i]))
        warm_time_slice = year_length.sel(year=slice(*warming_2[i]))
        warm_ann_wgts = warm_time_slice / warm_time_slice.sum()
        warm_ann_wgts = warm_ann_wgts.rename({'year' : 'time'})

        # get the weighted 30-year threshold statistics
        ann_warm_mean = wgt_ann_mean.sel(time=slice(*warming_2[i]))
        warm_wgtd = ann_warm_mean.weighted(warm_ann_wgts)
        warm_mean = warm_wgtd.mean("time")
        warm_mean.name = 'Mean'

        # other stats
        var_warm_median = var_warm.median("time")
        var_warm_median.name = 'Median'
        var_warm_max = var_warm.max("time")
        var_warm_max.name = 'Max'
        var_warm_min = var_warm.min("time")
        var_warm_min.name = 'Min'

        # make the warming threshold data set
        var_warm_stat = xr.merge([warm_mean,var_warm_median,var_warm_max,var_warm_min])

        for j,stat in enumerate(my_statistics):
            ax = axs[i,j]
            ax.set_ylabel(my_simulations[i])

            my_anom = (var_warm_stat.sel(simulation=sim).isel(scenario=0)[stat] -
                var_hist_stat.sel(simulation=sim).isel(scenario=0)[stat])

            cs = my_anom.plot(
                ax=ax, shading='auto', cmap="coolwarm", transform = ccrs.LambertConformal(),
                add_colorbar = False, vmin=-3,vmax=3)

            ax.coastlines(linewidth=1, color = 'black', zorder = 10) # Coastlines

            gl = ax.gridlines(linewidth=0.25, color='gray', alpha=0.9, crs=ccrs.PlateCarree(), 
                                  linestyle = '--',draw_labels=False)
            if (j==0):

                gl.left_labels = True
                ax.text(-0.6, 0.5,my_simulations[i], transform =ax.transAxes,
                        va='center', rotation='vertical',fontsize=12)

            if (i==0):
                ax.set_title(stat)

            else:
                ax.set_title('')

    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.01, top=0.92, left=0.1, right=.9,
                        wspace=0.1, hspace=0.1)

    # Add a colorbar axis at the right of the graph
    cbar_ax = fig.add_axes([.9, 0.33, 0.03, 0.25])

    # Draw the colorbar
    cbar=fig.colorbar(cs, cax=cbar_ax,orientation='vertical',
                     label='deg C')

    # fig.tight_layout() # not compatible with shared cbar

    fig.suptitle(var_str+ ' Anomalies for 2 deg Warming Level',y=1.)
    mpl_pane = pn.pane.Matplotlib(fig, dpi=144)
    display(mpl_pane)