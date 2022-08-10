from matplotlib.figure import Figure
import numpy as np 
import pandas as pd
import panel as pn
import pkg_resources 

# Import package data 
ssp119 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP1_1_9.csv')
ssp126 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP1_2_6.csv')
ssp245 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP2_4_5.csv')
ssp370 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP3_7_0.csv')
ssp585 = pkg_resources.resource_filename('climakitae', 'data/tas_global_SSP5_8_5.csv')
hist = pkg_resources.resource_filename('climakitae', 'data/tas_global_Historical.csv')


class GMTContextPlot(): 
   
    def view(self): 

        ## Read in data
        ssp119_data = pd.read_csv(ssp119, index_col='Year')
        ssp126_data = pd.read_csv(ssp126, index_col='Year')
        ssp245_data = pd.read_csv(ssp245, index_col='Year')
        ssp370_data = pd.read_csv(ssp370, index_col='Year')
        ssp585_data = pd.read_csv(ssp585, index_col='Year')
        hist_data = pd.read_csv(hist, index_col='Year')

        ## x-axes 
        hist_t = np.arange(1950,2015,1)
        cmip_t = np.arange(2015,2100,1)

        ## Colors for each line, matching IPCC standard colors 
        #https://pyam-iamc.readthedocs.io/en/stable/tutorials/ipcc_colors.html
        c119 = "#00a9cf"
        c126 = "#003466"
        c245 = "#f69320"
        c370 = "#df0000"
        c585 = "#980002"

        # Set up figure 
        fig = Figure(figsize=(8, 6))
        ax = fig.subplots()
        ax.set_ylim([-1,5])
        ax.set_xlim([1950,2100]);
        ax.set_xticks([1950,2000,2015,2100])
        ax.annotate("°C", xy=(-0.05, 1.05), xycoords='axes fraction')
        ax.grid(visible=True, which='major', axis='y', color='0.75')

        # CMIP6 mean lines
        ax.plot(hist_t, hist_data['Mean'], color='k') # very likely range
        ax.plot(cmip_t, ssp119_data['Mean'], c=c119)
        ax.plot(cmip_t, ssp126_data['Mean'], c=c126) # very likely range
        ax.plot(cmip_t, ssp245_data['Mean'], c=c245)
        ax.plot(cmip_t, ssp370_data['Mean'], c=c370) # very likely range
        ax.plot(cmip_t, ssp585_data['Mean'], c=c585)

        # Very likely ranges: 90-100%
        ax.fill_between(hist_t, hist_data['5%'], hist_data['95%'], color='k', alpha=0.1);
        ax.fill_between(cmip_t, ssp126_data['5%'], ssp126_data['95%'], color=c126, alpha=0.1);
        ax.fill_between(cmip_t, ssp370_data['5%'], ssp370_data['95%'], color=c370, alpha=0.1);

        # Labels on right hand side
        lidx = 2099 # last index of cmip6 dataframes
        f = 12 # fontsize for labels
        ax.annotate("SSP1-1.9", xy=(cmip_t[-1]+3, ssp119_data['Mean'][lidx]), xycoords='data', annotation_clip=False, c=c119, fontsize=f);
        ax.annotate("SSP1-2.6", xy=(cmip_t[-1]+3, ssp126_data['Mean'][lidx]), xycoords='data', annotation_clip=False, c=c126, fontsize=f);
        ax.annotate("SSP2-4.5", xy=(cmip_t[-1]+3, ssp245_data['Mean'][lidx]), xycoords='data', annotation_clip=False, c=c245, fontsize=f);
        ax.annotate("SSP3-7.0", xy=(cmip_t[-1]+3, ssp370_data['Mean'][lidx]), xycoords='data', annotation_clip=False, c=c370, fontsize=f);
        ax.annotate("SSP5-8.5", xy=(cmip_t[-1]+3, ssp585_data['Mean'][lidx]), xycoords='data', annotation_clip=False, c=c585, fontsize=f);

        # Title
        ax.set_title("Global surface temperature change relative to 1850-1900", y=1.05, fontsize=f+2);

        # Generate panel figure 
        mpl_pane = pn.pane.Matplotlib(fig, dpi=144)
        return mpl_pane 

def _display_warming_levels():
    GMT_plot = GMTContextPlot() 
    panel_doodad = pn.Column(GMT_plot.view())
    return panel_doodad