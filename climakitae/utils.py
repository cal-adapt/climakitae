import pandas as pd 

# Read csv file containing variable information as dictionary
def _read_var_csv(csv_file, index_col="name", usecols=["name","description","extended_description"]): 
    """Read in variable descriptions csv file as a dictionary
    
    Args: 
        csv_file (str): Local path to variable csv file 
        index_col (str): Column in csv to use as keys in dictionary
        
    Returns: 
        descrip_dict (dictionary): Dictionary containing index_col as keys and additional columns as values 
    
    """
    csv = pd.read_csv(csv_file, index_col=index_col, usecols=usecols)
    descrip_dict = csv.to_dict(orient="index")
    return descrip_dict