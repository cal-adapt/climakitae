"""generate_var_csv.py 
    Generate a csv from catalog variable names and descriptions 
"""

import csv 
import os
import sys
from climakitae.selectors import CatalogContents


def generate_var_csv(cat_contents, csv_filename): 
    """Generate csv using catalog contents. Input a string filename to give output csv. """
    header = ["name","description","extended_description"]
    with open(csv_filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header) # Write csv header (column names) 
        names = []
        for time_scale in ["hourly","daily"]:
            for model_type in ["Dynamical","Statistical"]: 
                try: 
                    catalog_iter = cat_contents._variable_choices[time_scale][model_type].items()
                except: 
                    pass # Don't raise error, just skip line 
                for row in catalog_iter:
                    try: 
                        name = row[1]
                        description = row[0]   
                        if name not in names: 
                            writer.writerow([name,description,""])
                            names.append(name)
                        else: 
                            pass # Don't repeat variable name 
                    except:
                        pass # Don't raise error, just skip line
                    
    return None 

                    
if __name__ == "__main__":
    cat_contents = CatalogContents()
    csv_filename = "variable_descriptions.csv"
    generate_var_csv(cat_contents=cat_contents, csv_filename=csv_filename)