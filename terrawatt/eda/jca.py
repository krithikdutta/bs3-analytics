import numpy as np
import pandas as pd
import libpysal
from esda.join_counts import Join_Counts
from shapely.geometry import box
import os

"""
This script performs Join Count statistics to test for spatial autocorrelation
in binary (0/1) data, reading geographic data from a CSV file.

1.  Define a function `analyze_joins_from_csv` that takes a file path and
    column names.
2.  Read a CSV containing bounding box coordinates (e.g., xmin, ymin, xmax, ymax)
    and a binary data column.
3.  Create polygon geometries from the bounding boxes.
4.  Define spatial relationships by generating a 'Queen' contiguity weights
    matrix from the polygons.
5.  Calculate the Join Count statistic on the binary data.
6.  Print and interpret the results to determine if events are clustered,
    dispersed, or random.
7.  A main block that demonstrates usage by first creating a sample CSV,
    then running the analysis on it.
"""

def analyze_joins_from_csv(csv_path, data_column, bbox_columns):
    """
    Reads a CSV with bounding boxes, calculates spatial weights, and performs
    Join Count analysis.

    Args:
        csv_path (str): The full path to the input CSV file.
        data_column (str): The name of the column containing the binary (0/1) data.
        bbox_columns (list): A list of 4 strings for the bounding box columns,
                             in the order: [xmin, ymin, xmax, ymax].
    """
    # --- 0. Input Validation ---
    if not isinstance(bbox_columns, list) or len(bbox_columns) != 4:
        print("Error: bbox_columns must be a list of 4 string names.")
        return

    # --- 1. Read Data and Create Geometries ---
    try:
        df = pd.read_csv(csv_path)
        # Create a list of Shapely Polygon objects from the bounding boxes
        # Uses the user-provided column names
        geometries = [box(row[bbox_columns[0]], row[bbox_columns[1]], row[bbox_columns[2]], row[bbox_columns[3]]) for index, row in df.iterrows()]
        print(f"Successfully read {len(df)} records from '{csv_path}'.")
    except (FileNotFoundError, KeyError, AttributeError) as e:
        print(f"Error reading or processing CSV: {e}")
        print(f"Please ensure the file exists and contains the specified columns: {bbox_columns}")
        return

    # Extract the binary data variable as a NumPy array
    y = df[data_column].values
    print(f"Analyzing data from column: '{data_column}'")
    print(f"Number of events (data=1): {int(np.sum(y))}\n")


    # --- 2. Create a Spatial Weights Matrix (W) ---
    # This matrix is generated directly from the polygon geometries.
    # Queen contiguity means neighbors touch at an edge or a corner.
    try:
        w = libpysal.weights.Queen.from_iterable(geometries)
        w.transform = 'b' # Ensure weights are binary (0/1)
        print("Successfully created a 'Queen' contiguity spatial weights matrix from geometries.")
        # Display info for the first polygon's neighbors
        print(f"Sample - Neighbors for polygon 0: {w.neighbors[0]}\n")
    except Exception as e:
        print(f"Error creating weights matrix: {e}")
        return


    # --- 3. Perform Join Count Analysis ---
    jc = Join_Counts(y, w)


    # --- 4. Print and Interpret the Results ---
    print("--- Join Count Statistics Results ---")
    print(f"Observed Black-Black (1-1 joins): {jc.bb}")
    print(f"Expected: {jc.expected}")
    print(f"Observed: {jc.crosstab}")
    print("-" * 35)
    print("Chi-Square Test for Join Counts:")
    print(f"  chi2: {jc.chi2:.4f}")
    print(f"  P-value: {jc.chi2_p:.4f}")
    print(f"  Simulated chi2 P-value: {jc.p_sim_chi2:.4f}")
    print("-" * 35)
    print("p-value based on permutations (one-sided) null: spatial randomness alternative: the observed bb is greater than under randomness")
    print("p_sim_bb:", jc.p_sim_bb)

