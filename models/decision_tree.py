import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

#Implementation of Decision Tree to determine course recommendation strength based on course eval data

# Reading data
project_root = Path(__file__).resolve().parent.parent
csv_path = project_root /'data' /'clean_data' /'new_data.csv'

full_data = pd.read_csv(csv_path)

# Splitting data


# Building + Training model


# Testing model


# Creating Visualization