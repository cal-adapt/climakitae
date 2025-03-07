"""
Util for generating estimated crossing years for global warming levels

Outputs a csv file for reference in ../data/ ###

To run, type: <<python generate_gwl_timing_table.py>> in the command line. Runs in about 20 seconds and returns no output.
"""

from climakitae.util.utils import read_csv_file, write_csv_file
import numpy as np
import pandas as pd
from climakitae.core.paths import (
    ssp119_file,
    ssp126_file,
    ssp245_file,
    ssp370_file,
    ssp585_file,
)


def main():
    """
    Generates global warming level (GWL) timing table based on the IPCC warming trajectories and ranges in ../data/

    Uses the same method that  ../explore/warming.py uses to generate visualizations to identify crossing years.

    Saves the mean, 5th, and 95th percentile for each SSP scenario

    """

    # load IPCC scenaario trajectories
    ssp119_data = read_csv_file(ssp119_file, index_col="Year")
    ssp126_data = read_csv_file(ssp126_file, index_col="Year")
    ssp245_data = read_csv_file(ssp245_file, index_col="Year")
    ssp370_data = read_csv_file(ssp370_file, index_col="Year")
    ssp585_data = read_csv_file(ssp585_file, index_col="Year")

    ssp_dict = {
        "SSP_1-1.9": ssp119_data,
        "SSP_1-2.6": ssp126_data,
        "SSP_2-4.5": ssp245_data,
        "SSP_3-7.0": ssp370_data,
        "SSP_5-8.5": ssp585_data,
    }

    # specify options
    warmlevels = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    cmip_t = np.arange(2015, 2100, 1)

    # construct dataframe
    cols = []
    for ssp in ssp_dict.keys():
        cols.extend([ssp + "_05_percentile", ssp + "_mean", ssp + "_95_percentile"])
    wl_timing_df = pd.DataFrame(columns=cols)

    # fill dataframe with crossing years
    for warmlevel in warmlevels:
        row = []
        for ssp in ssp_dict.keys():
            ssp_selected = ssp_dict[ssp]  # data selected

            # Only add data for a scenario if the mean and upper bound of uncertainty reach the gwl
            if (np.argmax(ssp_selected["Mean"] > warmlevel) > 0) and (
                (np.argmax(ssp_selected["95%"] > warmlevel)) > 0
            ):

                x_5 = cmip_t[0] + np.argmax(ssp_selected["95%"] > warmlevel)

                if np.argmax(ssp_selected["5%"] > warmlevel):
                    x_95 = cmip_t[0] + np.argmax(ssp_selected["5%"] > warmlevel)
                else:
                    # set year to 2100 if the lower bound of uncertainty does not hit the gwl
                    x_95 = 2100

                year_warmlevel_reached = (
                    ssp_selected.where(ssp_selected["Mean"] > warmlevel)
                    .dropna()
                    .index[0]
                )

            else:
                x_95 = np.nan
                x_5 = np.nan
                year_warmlevel_reached = np.nan

            row.extend([x_5, year_warmlevel_reached, x_95])

        wl_timing_df.loc[warmlevel] = row

    write_csv_file(wl_timing_df, "data/gwl_timing_table.csv")


if __name__ == "__main__":
    main()
