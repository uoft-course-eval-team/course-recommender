import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data/clean_data/new_data.csv")
df.columns = df.columns.str.strip().str.lower()
df["recommended"] = (df["i would recommend this course"] >= 4).astype(int)
df["description"] = df["description"].fillna("")
df[["course", "term", "last name"]] = df[["course", "term", "last name"]].fillna("missing")

# Debug
# print("Number of columns:", len(df.columns))
# print("Column names:", df.columns.tolist())

# Select relevant columns
numerical_features = [
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
categorical_features = ["course", "term", "last name"]

df = df.dropna(subset=numerical_features + ["recommended"])  # Drop missing value rows

X_num = df[numerical_features].reset_index(drop=True)  # Processing features
X_cat = pd.get_dummies(df[categorical_features], drop_first=True).reset_index(drop=True)

vectorizer = TfidfVectorizer(max_features=100, stop_words = 'english')  # Vectorizing course description
X_text = pd.DataFrame(vectorizer.fit_transform(df["description"]).toarray(), columns=vectorizer.get_feature_names_out())

X = pd.concat([X_num, X_cat, X_text], axis=1)  # Combining
y = df["recommended"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Train/test split

model = LinearRegression()
# Save the trained model 
joblib.dump(model, "models/saved_models/linear_regression.sav")
model.fit(X_train, y_train)

y_pred_score = model.predict(X_test)
y_pred = (y_pred_score >= 0.5).astype(int)

# Evaluation
print("Linear Regression (as classifier) Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Greens")
plt.title("Linear Regression Confusion Matrix")
plt.show()

# Coefficients
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
top_coef = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_coef.head(20), x="Coefficient", y="Feature", palette="crest")
plt.title("Top 20 Linear Regression Coefficients")
plt.axvline(0, color="black")
plt.tight_layout()
plt.show()
