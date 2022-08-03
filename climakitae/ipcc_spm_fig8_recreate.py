"""
This script reproduces the IPCC Summary for Policymakers Fig 8.
(https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf) for the Cal-Adapt: Analytics Engine Warming Levels
notebok in development. SPM Fig. 8 is a reproduction of Chapter 4 - Figure 4.2.

This figure will be one panel of several options, but this will fixed in as a reference point.
WGIII Code will be here eventually: https://www.ipcc-data.org/ar6landing.html
Data download: https://data.ceda.ac.uk/badc/ar6_wg1/data/spm/spm_08/v20210809/panel_a
"""

## Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Read in data
ssp119_data = pd.read_csv("tas_global_SSP1_1_9.csv", index_col='Year')
ssp119_mean = ssp119_data['Mean']
ssp126_data = pd.read_csv("tas_global_SSP1_2_6.csv", index_col='Year')
ssp126_mean = ssp126_data['Mean']
ssp126_low = ssp126_data['5%']
ssp126_high = ssp126_data['95%']
ssp245_data = pd.read_csv("tas_global_SSP2_4_5.csv", index_col='Year')
ssp245_mean = ssp245_data['Mean']
ssp370_data = pd.read_csv("tas_global_SSP3_7_0.csv", index_col='Year')
ssp370_mean = ssp370_data['Mean']
ssp370_low = ssp370_data['5%']
ssp370_high = ssp370_data['95%']
ssp585_data = pd.read_csv("tas_global_SSP5_8_5.csv", index_col='Year')
ssp585_mean = ssp585_data['Mean']
hist_data = pd.read_csv("tas_global_Historical.csv", index_col='Year')
hist_mean = hist_data['Mean']
hist_low = hist_data['5%']
hist_high = hist_data['95%']

## Plot figure
hist_t = np.arange(1950,2015,1)
cmip_t = np.arange(2015,2100,1)

## https://pyam-iamc.readthedocs.io/en/stable/tutorials/ipcc_colors.html
c119 = "#00a9cf"
c126 = "#003466"
c245 = "#f69320"
c370 = "#df0000"
c585 = "#980002"

# Figure set-up
fig = plt.figure(figsize = (7,4))
plt.ylim([-1,5])
plt.xlim([1950,2100]);
plt.xticks([1950,1960,1970,1980,1990,2000,2010,2015,2020,2030,2040,2050,2060,2070,2080,2090,2100],
           labels=["1950","","","","","2000","","2015","","","","2050","","","","","2100"]);
plt.annotate("°C", xy=(-0.05, 1.05), xycoords='axes fraction');
f = 12 # fontsize for labels

# CMIP6 mean lines
plt.plot(hist_t, hist_mean, color='k'); # very likely range
plt.plot(cmip_t, ssp119_mean, c=c119);
plt.plot(cmip_t, ssp126_mean, c=c126); # very likely range
plt.plot(cmip_t, ssp245_mean, c=c245);
plt.plot(cmip_t, ssp370_mean, c=c370); # very likely range
plt.plot(cmip_t, ssp585_mean, c=c585);

# Very likely ranges: 90-100%
plt.fill_between(hist_t, hist_low, hist_high, color='k', alpha=0.1);
plt.fill_between(cmip_t, ssp126_low, ssp126_high, color=c126, alpha=0.1);
plt.fill_between(cmip_t, ssp370_low, ssp370_high, color=c370, alpha=0.1);

# Labels on right hand side
lidx = 2099 # last index of cmip6 dataframes
plt.annotate("SSP1-1.9", xy=(cmip_t[-1]+3, ssp119_mean[lidx]), xycoords='data', annotation_clip=False, c=c119, fontsize=f);
plt.annotate("SSP1-2.6", xy=(cmip_t[-1]+3, ssp126_mean[lidx]), xycoords='data', annotation_clip=False, c=c126, fontsize=f);
plt.annotate("SSP2-4.5", xy=(cmip_t[-1]+3, ssp245_mean[lidx]), xycoords='data', annotation_clip=False, c=c245, fontsize=f);
plt.annotate("SSP3-7.0", xy=(cmip_t[-1]+3, ssp370_mean[lidx]), xycoords='data', annotation_clip=False, c=c370, fontsize=f);
plt.annotate("SSP5-8.5", xy=(cmip_t[-1]+3, ssp585_mean[lidx]), xycoords='data', annotation_clip=False, c=c585, fontsize=f);

# Title
plt.title("Global surface temperature change relative to 1850-1900", x=-0.05, y=1.1, loc='left', fontsize=f+2);

# 3°C exceedance lines
# plt.grid(visible=True, which='major', axis='y', color='0.75')
warmlevel = 3.0
plt.axhline(y=warmlevel, color='k', lw=1);

means = [ssp119_mean, ssp126_mean, ssp245_mean, ssp370_mean, ssp585_mean]
for i in means:
    if np.argmax(i > warmlevel) != 0:
        plt.axvline(x=cmip_t[0] + np.argmax(i > warmlevel), color='k', linestyle='--', lw=1);

fig.savefig("spm_fig8.jpeg", dpi=300, bbox_inches='tight');
