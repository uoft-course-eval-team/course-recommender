import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("new_data.csv")
df.columns = df.columns.str.strip().str.lower()

# Debug
# print("Number of columns:", len(df.columns))
# print("Column names:", df.columns.tolist())

# Select relevant columns
feature_cols = [
    "year",
    "item 1 (i found the course intellectually stimulating)",
    "item 2 (the course provided me with a deep understanding of the subject manner)",
    "item 3 (the instructor created a course atmosphere that was condusive to my learning)",
    "item 4 (course projects, assignments, tests, and/or exams improved my understanding of the course material)",
    "item 5 (course projects, assignments, tests, and/or exams provided opportunity for me to demonstrate an "
    "understanding of the course material)",
    "item 6 (overall, the quality of my learning experience in the course was:)",
    "instructor generated enthusiasm",
    "course workload"
]
target_col = "i would recommend this course"

df = df.dropna(subset=feature_cols + [target_col])  # Drop missing value rows

X = df[feature_cols]  # Extract features
y = df[target_col]  # extract target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Train/test split

model = LinearRegression()
model.fit(X_train, y_train)
y_pred_score = model.predict(X_test)

y_pred_binary = (y_pred_score >= 4).astype(int)  # Threshold at 4
y_test_binary = (y_test >= 4).astype(int)

# Evaluation
print("Linear Regression as Classifier Accuracy:", accuracy_score(y_test_binary, y_pred_binary))
print(classification_report(y_test_binary, y_pred_binary))

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test_binary, y_pred_binary, cmap="Greens")
plt.title("Linear Regression Thresholded (≥4) Confusion Matrix")
plt.show()

# Coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_.flatten()
}).sort_values(by="Coefficient", key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=coef_df.head(20), x="Coefficient", y="Feature", palette="viridis")
plt.title("Top 20 Linear Regression Coefficients")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()
