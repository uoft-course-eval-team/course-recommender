import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

"""
This file is a preliminary exploration of the data used in our project for the purpose of understanding its shape
and key characteristics
"""

# Reading data
project_root = Path(__file__).resolve().parent
csv_path = project_root /'data' /'clean_data' /'new_data.csv'

dataset = pd.read_csv(csv_path)

# Basic Overview
print("\n")
print("Dataset Info")
print(dataset.info())

print("\n")
print("Dataset Shape")
print(f"Rows: {dataset.shape[0]}, Columns: {dataset.shape[1]}")

print("\n")
print("Dataset Summary Statistics")
print(dataset.describe(include='all'))
print("\n")

# Missing Values (might get rid of this one)

missing_vals = (
    dataset.isna().sum()
    .reset_index()
    .rename(columns={"index": "Column", 0: "MissingCount"}))

missing_vals["MissingPercent"] = 100 * missing_vals["MissingCount"] / len(dataset)
missing_vals = missing_vals.sort_values("MissingCount", ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(x="MissingPercent", y="Column", data=missing_vals)
plt.title("Missing Values (%)")
#plt.tight_layout()
path_missing = project_root / 'graphs' / 'missing_values.png'
plt.savefig(path_missing)

#Recommend this course

plt.figure(figsize=(8, 4))
plt.hist(x=dataset["i would recommend this course"], bins=5)
plt.title("Histogram of Numeric Recommendation Value")
plt.tight_layout()
path_targ_hist = project_root / 'graphs' / 'target_hist.png'
plt.savefig(path_targ_hist)

plt.figure(figsize=(8, 4))
sns.countplot(x="recommended", data=dataset)
plt.title("Distribution of Recommendations")
plt.tight_layout()
path_targ_cnt = project_root / 'graphs' / 'target_cnt.png'
plt.savefig(path_targ_cnt)

# Scatterplots

numerical_features = (dataset.select_dtypes(include=["number"]).drop(columns=['i would recommend this course'])
                      .columns.tolist())
categorical_features = (dataset.select_dtypes(exclude=["number"]).drop(columns=['division', 'dept',
                                                                         'description']).columns.tolist())

# scuffed
df_long = dataset.melt(id_vars=['i would recommend this course'], value_vars=numerical_features,
                  var_name='feature', value_name='feature_value')

g = sns.relplot(
    data=df_long, x='feature_value', y='i would recommend this course',
                                        col='feature',col_wrap=3, kind='scatter', height=4,aspect=1)
g.set_axis_labels("Feature Value", 'i would recommend this course')
g.set_titles("{col_name}")
plt.title("Pair plot of Numerical Recommendation")
path_scatter = project_root / 'graphs' / 'scatter.png'
plt.savefig(path_scatter)

