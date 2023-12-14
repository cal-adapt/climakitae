"""Backend for agnostic tools."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_lookup_table():
    """
    Create a warming level table that is subsetting only on GCMs that we have data for.
    """
    # Finding the names of all the GCMs that we catalog
    from climakitae.core.data_interface import DataInterface
    data_interface = DataInterface()
    gcms = data_interface.data_catalog.df.source_id.unique()
    
    # Reading GCM warming levels 1850-1900 table
    temp_df = pd.read_csv('~/src/climakitae/climakitae/data/gwl_1850-1900ref_1to4deg_per05.csv')
    # Clean long float column names
    temp_df.columns = np.append(
        temp_df.columns[:3], 
        np.round(temp_df.columns[3:].astype(float), 2).astype(str)
    )
    
    # Subsetting on the table for only the GCMs that we track
    df = temp_df[temp_df['GCM'].isin(gcms)]
    return df


def warm_level_to_years(warm_df, scenario, warming_level):
    """Given warming level, plot histogram of years and label median year."""
    datetimes = warm_df[warm_df['scenario'] == scenario][warming_level].astype('datetime64[ns]')
    years = datetimes.dt.year
    med_datetime = datetimes.quantile(0.5, interpolation='midpoint')
    med_year = med_datetime.year
    fig, ax = plt.subplots()
    plot_years(ax, years, med_year)
    return None


def plot_years(ax, years, med_year):
    """Plot histogram of years and label median year, on axis."""
    n = len(years)
    ax.hist(years)
    ax.set_title(f'Years when each of all {n} model simulations reach the warming level')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of simulations')
    ax.axvline(med_year, color='black', linestyle='--', label=f'Median year is {med_year}')
    ax.legend()
    return ax


def warm_level_to_year(warm_df, scenario, warming_level):
    """Given warming level, give year."""
    med_date = warm_df[warm_df['scenario'] == scenario][warming_level].astype('datetime64[ns]').quantile(0.5, interpolation='midpoint')
    return med_date.year


def warm_level_to_month(warm_df, scenario, warming_level):
    """Given warming level, give month."""
    med_date = warm_df[warm_df['scenario'] == scenario][warming_level].astype('datetime64[ns]').quantile(0.5, interpolation='midpoint')
    return med_date.strftime('%Y-%m')


def plot_warm_levels(fig, ax, levels, med_level):
    """Plot histogram of warming levels, on axis."""
    n = len(levels)
    ax.hist(levels)
    fig.suptitle(f'Warming levels reached in the year by {n} model simulations')
    ax.set_title('(out of 80 simulations)', fontsize=10)
    ax.set_xlabel('Warming level (°C)')
    ax.set_ylabel('Number of simulations')
    # ax.axvline(
    #     med_level, color='black', linestyle='--', 
    #     label=f'Median warming level is {med_level}'
    # )
    # ax.legend()
    return ax


def year_to_warm_levels(warm_df, scenario, year):
    """Given year, give warming levels and their median."""
    warm_levels = np.arange(1, 4.01, .05).round(2).astype(str)
    date_df = warm_df[warm_df['scenario'] == scenario][warm_levels]

    # Creating new counts dataframe
    counts_df = pd.DataFrame()
    years = pd.to_datetime(date_df.values.flatten()).year
    years_idx = set(years[pd.notna(years)].sort_values())
    counts_df.index = years_idx

    # Creating counts by warming level and year
    for level in warm_levels:
        counts_df = counts_df.merge(pd.to_datetime(date_df[level]).dt.year.value_counts(), left_index=True, right_index=True, how='outer')
        counts_df = counts_df.fillna(0)   
    counts_df = counts_df.T

    # Find the warming levels and their median
    levels = counts_df[year]
    expanded_levels = [
        float(value) for value, count in levels.items() for _ in range(int(count))
    ]
    med_level = np.quantile(
        expanded_levels, 0.5, interpolation='nearest'
    )
    return expanded_levels, med_level


def round_to_nearest_half(number):
    # TODO: what to do with .25
    return round(number * 2) / 2


def find_warm_index(warm_df, scenario='ssp370', warming_level=None, year=None):
    """
    Given a scenario and either a warming level or a time, return either the median warming level or month from all simulations within this scenario.
    
    Note the median warming level is rounded to the nearest half.
    
    Parameters
    ----------
    scenario: string ('ssp370')
    warming_level: string ('1.5', '2.0', '3.0')
    year: int
        Must be in [2001, 2090].

    """
    if not warming_level and not year:
        print("Pass in either a warming level or a year.")
        
    elif warming_level is not None and year is not None:
        print("Pass in either a warming level or a year, but not both.")
        
    else:
        if scenario != 'ssp370':
            raise NotImplementedError(
                'Scenarios other than ssp370 are under development.'
            )

        # Given warming level, plot years and find month
        if warming_level is not None and year is None:
            allowed_warm_level = ['1.5', '2.0', '3.0']
            if warming_level not in allowed_warm_level:
                raise NotImplementedError(
                    f'Please choose a warming level among {allowed_warm_level}'
                )
            warm_level_to_years(warm_df, scenario, warming_level)
            return warm_level_to_month(warm_df, scenario, warming_level)

        # Given year, plot warming levels and find median
        elif warming_level is None and year is not None:
            min_year, max_year = 2001, 2090  # years with 10+ simulations
            if not (min_year <= year and year <= max_year):
                raise ValueError(
                    f'Please provide a year between {min_year} and {max_year}'
                )
            warm_levels, med_level = year_to_warm_levels(warm_df, scenario, year)
            fig, ax = plt.subplots()
            plot_warm_levels(fig, ax, warm_levels, med_level)
            return print(
                f'The median projected warming level is about {round_to_nearest_half(med_level)}°C \n'
            )
        

# Lambda function to pass in a created warming level table rather than remaking it with every call.
def create_conversion_function(warm_df):
    return lambda scenario='ssp370', warming_level=None, year=None: find_warm_index(warm_df, scenario, warming_level, year)

