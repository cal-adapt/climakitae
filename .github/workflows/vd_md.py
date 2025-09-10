import sys
import numpy as np
import pandas as pd
from tabulate import tabulate

def process_arguments(arg1, arg2):
    print(f"Input variable_descriptions csv path: {arg1}")
    print(f"Output markdown file path: {arg2}")

def vd_csv_to_markdown(csv_file_path: str, output_markdown_path: str):
    """Converts variable_descriptions CSV file to an enhanced Markdown table.

    Parameters
    ----------
        csv_file_path: str
            The path to the input variable CSV file.
        output_markdown_path: str, optional
            The path to save the output Markdown file.
            If None, the Markdown table is printed to the console.
    """
    try:
        # Load the variable descriptions CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)

        df.insert(
            1,
            "derived",
            np.where(df["variable_id"].str.endswith("_derived"), "True", "False"),
        )
        # Convert the DataFrame to a Markdown table string
        # 'github' formssat creates a standard Markdown table with GitHub format
        # 'headers="keys"' uses column names from the DataFrame as table headers
        markdown_table = tabulate(
            df, headers="keys", tablefmt="github", showindex=False
        )

        if output_markdown_path:
            with open(output_markdown_path, "w") as f:
                f.write(markdown_table)
            print(f"Markdown table saved to: {output_markdown_path}")
        else:
            print(markdown_table)

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 2:  # Check if two arguments are provided
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]

        process_arguments(arg1, arg2)
        vd_csv_to_markdown(arg1, arg2)
    else:
        print("Usage: python vd_md.py <input_csv> <output_md>")
