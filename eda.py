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
dataset.rename(columns={
    'item 1 (i found the course intellectually stimulating)': 'item 1',
    'item 2 (the course provided me with a deep understanding of the subject manner)': 'item 2',
    'item 3 (the instructor created a course atmosphere that was condusive to my learning)': 'item 3',
    'item 4 (course projects, assignments, tests, and/or exams improved my understanding of the course material)': 'item 4',
    'item 5 (course projects, assignments, tests, '
    'and/or exams provided opportunity for me to demonstrate an understanding of the course material)': 'item 5',
    'item 6 (overall, the quality of my learning experience in the course was:)': 'item 6'
}, inplace=True)

# Basic Overview
print("\n")
print("Dataset Info")
print(dataset.drop(columns=['dept', 'division', ]).info())

print("\n")
print("Dataset Shape")
print(f"Rows: {dataset.shape[0]}, Columns: {dataset.shape[1]}")

print("\n")
print("Dataset Summary Statistics")
print(dataset.drop(columns=['dept', 'division', ]).describe(include='all'))
print("\n")



#-----------------------------------------------------

def plot_dataset_info(df, filepath):
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Non-null count': df.notnull().sum(),
        'Dtype': df.dtypes.astype(str)
    })

    fig, ax = plt.subplots(figsize=(8, len(df.columns)*0.5))
    ax.axis('off')

    table_data = info_df.values
    col_labels = info_df.columns.tolist()

    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.title('Dataset Info: Non-null counts and Data Types', fontsize=14)
    plt.tight_layout()
    plot_path = project_root /'graphs' / filepath
    plt.savefig(plot_path)
    plt.close()

def plot_dataset_shape(df, filepath):
    shape_data = {'Rows': df.shape[0], 'Columns': df.shape[1]}
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(shape_data.keys(), shape_data.values(), color=['skyblue', 'salmon'])
    ax.set_ylabel('Count')
    ax.set_title('Dataset Shape')
    for i, v in enumerate(shape_data.values()):
        ax.text(i, v + max(shape_data.values()) * 0.02, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plot_path = project_root / 'graphs' / filepath
    plt.savefig(plot_path)
    plt.close()

def plot_summary_stats(df, filepath):
    summary = df.describe().T  # Numeric summary stats
    # Select subset of stats to plot
    stats_to_plot = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']

    summary = summary[stats_to_plot]

    plt.figure(figsize=(10, len(summary)*0.5))
    sns.heatmap(summary, annot=True, cmap=sns.color_palette(["#4C72B0"]), cbar=False, fmt=".2f")
    plt.title('Numerical Features Summaries')
    plt.tight_layout()
    plot_path = project_root / 'graphs' / filepath
    plt.savefig(plot_path)
    plt.close()

plot_dataset_info(dataset, 'dataset_info.png')
plot_dataset_shape(dataset, 'dataset_shape.png')
plot_summary_stats(dataset, 'summary_stats.png')

# Categorical Summary Stats
categorical_features = dataset.select_dtypes(exclude=["number"])
cat_summary = pd.DataFrame({
    'Unique Values': categorical_features.nunique(),
    'Most Frequent': categorical_features.mode().iloc[0],
    'Frequency': categorical_features.apply(lambda x: x.value_counts().iloc[0])
})

plt.figure(figsize=(8, len(cat_summary) * 0.5 + 1))
sns.heatmap(cat_summary[['Unique Values', 'Frequency']],
            annot=True, fmt='g', cmap=sns.color_palette(["#66c2a5"]),
            cbar=False)
plt.title("Categorical Variable Summary")
plt.tight_layout()
cat_path = project_root / 'graphs' / 'categorical_summaries.png'
plt.savefig(cat_path)

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
"""df_long = dataset.melt(id_vars=['i would recommend this course'], value_vars=numerical_features,
                  var_name='feature', value_name='feature_value')

g = sns.relplot(
    data=df_long, x='feature_value', y='i would recommend this course',
                                        col='feature',col_wrap=3, kind='scatter', height=4,aspect=1)
g.set_axis_labels("Feature Value", 'i would recommend this course')
g.set_titles("{col_name}")
plt.title("Pair plot of Numerical Recommendation")
path_scatter = project_root / 'graphs' / 'scatter.png'
plt.savefig(path_scatter)"""


