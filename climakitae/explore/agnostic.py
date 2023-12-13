"""Backend for agnostic tools."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    """Plot histogram of years and label median year on axis."""
    ax.hist(years)
    ax.set_title('Years when different model simulations reach the warming level')
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
    """Given warming level, give year and month."""
    med_date = warm_df[warm_df['scenario'] == scenario][warming_level].astype('datetime64[ns]').quantile(0.5, interpolation='midpoint')
    return med_date.strftime('%Y-%m')


def find_warm_index(warm_df, scenario, warming_level=None, year=None):
    """
    Given a scenario and either a warming level or a year, return either the median warming level or year from all simulations within this scenario.
    
    Parameters
    ----------
    scenario: string ('ssp245', 'ssp370', 'ssp585')
    warming_level: string ('1.5', '2.0', '3.0', '4.0')
    year: int
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

        if warming_level is not None and year is None:
            allowed_warm_level = ['1.5', '2.0', '3.0']
            if warming_level not in allowed_warm_level:
                raise NotImplementedError(
                    f'Please choose a warming level among {allowed_warm_level}'
                )
            warm_level_to_years(warm_df, scenario, warming_level)
            return warm_level_to_month(warm_df, scenario, warming_level)

        elif warming_level is None and year is not None:
            
            # Given year, find warming level
            warm_levels = ['1.5', '2.0', '3.0', '4.0']
            date_df = warm_df[warm_df['scenario'] == 'ssp370'][warm_levels]
    
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
    
            # Find the median warming level for ssp370 and 2040.
            levels = counts_df[year]
            med_level = np.median([float(value) for value, count in levels.items() for _ in range(int(count))])
            return med_level
        

# Lambda function to pass in a created warming level table rather than remaking it with every call.

# TODO: Check out 2050, 2060, check out 2030
def pass_in_warm_df(warm_df):
    return lambda scenario, warming_level=None, year=None: find_warm_index(warm_df, scenario, warming_level, year)

