def compute_total_precip(cumulus_precip, gridcell_precip, variable_name="TOT_PRECIP"): 
    """ Compute total precipitation 
    
    Args: 
        cumulus_precip (xr.DataArray): Accumulated total cumulus precipitation (mm)
        gridcell_precip (xr.DataArray): Accumulated total grid scale precipitation (mm) 
        
    Returns: 
        total_precip (xr.DataArray): Total precipitation (mm)
    """
    
    total_precip = cumulus_precip + gridcell_precip 
    exists_negatives = (total_precip < 0).any().values.item()
    if exists_negatives: # Check for negative values 
        raise ValueError("Total precipitation must be a positive value. Your computation has returned a negative value.") 
    total_precip.name = variable_name
    return total_precip