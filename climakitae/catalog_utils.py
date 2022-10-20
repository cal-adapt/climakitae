"""Helper functions for working with ESM catalog and user data selections"""

def _convert_resolution(resolution, reverse=False):
    """Convert resolution format to grid_label format matching catalog names.
    Set reverse=True to get resolution format from input grid_label.
    """
    res_dict = {"45km":"d01", "9km":"d02", "3km":"d03"}

    if reverse == True:
        res_dict = {v: k for k, v in res_dict.items()}

    return res_dict[resolution]

def _convert_timescale(timescale, reverse=False):
    """Convert resolution format to table_id format matching catalog names.
    Set reverse=True to get resolution format from input table_id.
    """
    timescale_dict = {"monthly":"mon", "daily":"day", "hourly":"1hr"}

    if reverse == True:
        timescale_dict = {v: k for k, v in timescale_dict.items()}

    return timescale_dict[timescale]

def _convert_scenario(scenario, reverse=False):
    """
    Convert scenario format to experiment_id format matching catalog names.
    Set reverse=True to get scenario format from input experiement_id.
    """
    scenario_dict = {
        "Historical Reconstruction":"reanalysis",
        "Historical Climate":"historical", 
        "SSP 2-4.5 -- Middle of the Road":"ssp245",
        "SSP 5-8.5 -- Burn it All":"ssp585",
        "SSP 3-7.0 -- Business as Usual":"ssp370"
    }

    if reverse == True:
        scenario_dict = {v: k for k, v in scenario_dict.items()}

    return scenario_dict[scenario]
