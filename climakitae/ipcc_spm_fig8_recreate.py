"""
This script reproduces the IPCC Summary for Policymakers Fig 8 from AR6.
(https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf) for the Cal-Adapt: Analytics Engine Warming Levels
notebok in development. SPM Fig. 8 is a summary figure of Chapter 4 - Figure 4.2.

This figure will be one panel of several options, but will be a fixed reference point.
WGIII Code will be here: https://www.ipcc-data.org/ar6landing.html
Data access: https://data.ceda.ac.uk/badc/ar6_wg1/data/spm/spm_08/v20210809/panel_a

Data citation: Fyfe, J.; Fox-Kemper, B.; Kopp, R.; Garner, G. (2021):
Summary for Policymakers of the Working Group I Contribution to the IPCC Sixth Assessment Report - data for Figure SPM.8 (v20210809).
NERC EDS Centre for Environmental Data Analysis, 09 August 2021. doi:10.5285/98af2184e13e4b91893ab72f301790db.

IPCC AR6 citation: IPCC, 2021: Summary for Policymakers. In: Climate Change 2021:
The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change
[Masson-Delmotte, V., P. Zhai, A. Pirani, S.L. Connors, C. Péan, S. Berger, N. Caud, Y. Chen, L. Goldfarb, M.I. Gomis, M. Huang, K. Leitzell,
E. Lonnoy, J.B.R. Matthews, T.K. Maycock, T. Waterfield, O. Yelekçi, R. Yu, and B. Zhou (eds.)].
Cambridge University Press, Cambridge, United Kingdom and New York, NY, USA, pp. 3−32, doi:10.1017/9781009157896.001.
"""


## Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datadir = "climakitae/data/"

## Read in data
ssp119_data = pd.read_csv(datadir + "tas_global_SSP1_1_9.csv", index_col='Year')
ssp126_data = pd.read_csv(datadir + "tas_global_SSP1_2_6.csv", index_col='Year')
ssp245_data = pd.read_csv(datadir + "tas_global_SSP2_4_5.csv", index_col='Year')
ssp370_data = pd.read_csv(datadir + "tas_global_SSP3_7_0.csv", index_col='Year')
ssp585_data = pd.read_csv(datadir + "tas_global_SSP5_8_5.csv", index_col='Year')
hist_data = pd.read_csv(datadir + "tas_global_Historical.csv", index_col='Year')

## Plot figure
hist_t = np.arange(1950,2015,1)
cmip_t = np.arange(2015,2100,1)

## https://pyam-iamc.readthedocs.io/en/stable/tutorials/ipcc_colors.html
c119 = "#00a9cf"
c126 = "#003466"
c245 = "#f69320"
c370 = "#df0000"
c585 = "#980002"

ipcc_data = (hist_data.hvplot(y="Mean", color="k", label="Historical") *
             hist_data.hvplot.area(x="Year", y="5%", y2="95%", alpha=0.1, color="k", ylabel="°C", xlabel="", ylim=[-1,5], xlim=[1950,2100]) * # very likely range
             ssp119_data.hvplot(y="Mean", color=c119, label="SSP1-1.9") *
             # ssp126_data.hvplot.area(x="Year", y="5%", y2="95%", alpha=0.1, color=c126) * # very likely range
             ssp126_data.hvplot(y="Mean", color=c126, label="SSP1-2.6") *
             ssp245_data.hvplot(y="Mean", color=c245, label="SSP2-4.5") *
             ssp370_data.hvplot.area(x="Year", y="5%", y2="95%", alpha=0.1, color=c370) * # very likely range
             ssp370_data.hvplot(y="Mean", color=c370, label="SSP3-7.0") *
             ssp585_data.hvplot(y="Mean", color=c585, label="SSP5-8.5")
            )

# SSP intersection lines
cmip_t = np.arange(2015,2101,1)

# warming level connection lines & additional labeling
warmlevel = 2.5 ## this should not be hardcoded in, but provided as options
#warmlevel = [1.5, 2.0, 3.0, 4.0]
warmlevel_line = hv.HLine(warmlevel).opts(color="black", line_width=1.0) * hv.Text(x=1964, y=warmlevel+0.25, text=str(warmlevel) + "°C warming level").opts(style=dict(text_font_size='8pt'))

## Specifically only for SSP3-7.0 at present -- ahead of WG4
# If the mean/upperbound/lowerbound does not cross threshold, set to 2100 (not visible)
if (np.argmax(ssp370_data["Mean"] > warmlevel)) > 0:
    ssp370_int = hv.VLine(cmip_t[0] + np.argmax(ssp370_data["Mean"] > warmlevel)).opts(color=c370, line_dash="dashed", line_width=1)
else:
    ssp370_int = hv.VLine(cmip_t[0] + 2100).opts(color=c370, line_dash="dashed", line_width=1)

if (np.argmax(ssp370_data["95%"] > warmlevel)) > 0:
    ssp370_firstdate = hv.VLine(cmip_t[0] + np.argmax(ssp370_data["95%"] > warmlevel)).opts(color=c370,  line_width=1)
else:
    ssp370_firstdate = hv.VLine(cmip_t[0] + 2100).opts(color=c370,  line_width=1)

if (np.argmax(ssp370_data["5%"] > warmlevel)) > 0:
    ssp370_lastdate = hv.VLine(cmip_t[0] + np.argmax(ssp370_data["5%"] > warmlevel)).opts(color=c370,  line_width=1)
else:
    ssp370_lastdate = hv.VLine(cmip_t[0] + 2100).opts(color=c370, line_width=1)


## Bar to connect firstdate and lastdate of threshold cross
bar_y = -0.5
yr_len = [(cmip_t[0] + np.argmax(ssp370_data["95%"] > warmlevel), bar_y), (cmip_t[0] + np.argmax(ssp370_data["5%"] > warmlevel), bar_y)]
yr_rng = (np.argmax(ssp370_data["5%"] > warmlevel) - np.argmax(ssp370_data["95%"] > warmlevel))
if yr_rng > 0:
    interval = hv.Path(yr_len).opts(color=c370, line_width=1) * hv.Text(x=cmip_t[0] + np.argmax(ssp370_data["95%"] > warmlevel)+5,
                                                                    y=bar_y+0.25,
                                                                    text = str(yr_rng) + " yrs").opts(style=dict(text_font_size='8pt'))
else: # Removes "bar" in case the upperbound is beyond 2100
    interval = hv.Path([(0,0), (0,0)]) # hardcoding for now, likely a better way to handle

to_plot = ipcc_data * warmlevel_line * ssp370_int * ssp370_lastdate * ssp370_firstdate * interval
to_plot.opts(opts.Overlay(title='Global surface temperature change relative to 1850-1900', fontsize=12))
to_plot.opts(legend_position='bottom', fontsize=10)

#### -------------------------------------------------------------------------------------------------
#### The below code is for a static figure
# Figure set-up
# fig = plt.figure(figsize = (7,4))
# plt.ylim([-1,5])
# plt.xlim([1950,2100]);
# plt.xticks([1950,1960,1970,1980,1990,2000,2010,2015,2020,2030,2040,2050,2060,2070,2080,2090,2100],
#            labels=["1950","","","","","2000","","2015","","","","2050","","","","","2100"]);
# plt.annotate("°C", xy=(-0.05, 1.05), xycoords='axes fraction');
# plt.grid(visible=True, which='major', axis='y', color='0.75')
# f = 12 # fontsize for labels
#
# # CMIP6 mean lines
# plt.plot(hist_t, hist_data['Mean'], color='k'); # very likely range
# plt.plot(cmip_t, ssp119_data['Mean'], c=c119);
# plt.plot(cmip_t, ssp126_data['Mean'], c=c126); # very likely range
# plt.plot(cmip_t, ssp245_data['Mean'], c=c245);
# plt.plot(cmip_t, ssp370_data['Mean'], c=c370); # very likely range
# plt.plot(cmip_t, ssp585_data['Mean'], c=c585);
#
# # Very likely ranges: 90-100%
# plt.fill_between(hist_t, hist_data['5%'], hist_data['95%'], color='k', alpha=0.1);
# plt.fill_between(cmip_t, ssp126_data['5%'], ssp126_data['95%'], color=c126, alpha=0.1);
# plt.fill_between(cmip_t, ssp370_data['5%'], ssp370_data['95%'], color=c370, alpha=0.1);
#
# # Labels on right hand side
# lidx = 2099 # last index of cmip6 dataframes
# plt.annotate("SSP1-1.9", xy=(cmip_t[-1]+3, ssp119_data['Mean'][lidx]), xycoords='data', annotation_clip=False, c=c119, fontsize=f);
# plt.annotate("SSP1-2.6", xy=(cmip_t[-1]+3, ssp126_data['Mean'][lidx]), xycoords='data', annotation_clip=False, c=c126, fontsize=f);
# plt.annotate("SSP2-4.5", xy=(cmip_t[-1]+3, ssp245_data['Mean'][lidx]), xycoords='data', annotation_clip=False, c=c245, fontsize=f);
# plt.annotate("SSP3-7.0", xy=(cmip_t[-1]+3, ssp370_data['Mean'][lidx]), xycoords='data', annotation_clip=False, c=c370, fontsize=f);
# plt.annotate("SSP5-8.5", xy=(cmip_t[-1]+3, ssp585_data['Mean'][lidx]), xycoords='data', annotation_clip=False, c=c585, fontsize=f);
#
# # Title
# plt.title("Global surface temperature change relative to 1850-1900", x=-0.05, y=1.1, loc='left', fontsize=f+2);
#
# ## 3°C connection lines
# # plt.grid(visible=True, which='major', axis='y', color='0.75')     # gridlines at the whole degree mark
# warmlevel = 3.0
# plt.axhline(y=warmlevel, color='k', lw=1);
#
# means = [ssp119_data['Mean'], ssp126_data['Mean'], ssp245_data['Mean'], ssp370_data['Mean'], ssp585_data['Mean']]
# for i in means:
#     if np.argmax(i > warmlevel) != 0:
#         plt.axvline(x=cmip_t[0] + np.argmax(i > warmlevel), color='k', linestyle='--', lw=1);
#
# fig.savefig("spm_fig8.jpeg", dpi=300, bbox_inches='tight');
