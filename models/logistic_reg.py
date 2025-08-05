import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data/clean_data/new_data.csv")
df.columns = df.columns.str.strip().str.lower()
df["recommended"] = (df["i would recommend this course"] >= 4).astype(int)
df["description"] = df["description"].fillna("")
df[["course", "term", "last name"]] = df[["course", "term", "last name"]].fillna("missing")

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

df = df.dropna(subset=numerical_features + ["recommended"])  # Drop missing target

X_num = df[numerical_features].reset_index(drop=True)  # Process features
X_cat = pd.get_dummies(df[categorical_features], drop_first=True).reset_index(drop=True)

vectorizer = TfidfVectorizer(max_features=100, stop_words = 'english')  # Vectorize course description
X_text = pd.DataFrame(vectorizer.fit_transform(df["description"]).toarray(), columns=vectorizer.get_feature_names_out())

X = pd.concat([X_num, X_cat, X_text], axis=1)  # Combine
y = df["recommended"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split

model = LogisticRegression(max_iter=1000)  # Training model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)  # Predict

# Evaluate
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Save the trained model 
joblib.dump(model, "models/saved_models/logistic_regression.sav")

# Coefficients
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_.flatten()})
top_coef = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_coef.head(20), x="Coefficient", y="Feature", palette="coolwarm")
plt.title("Top 20 Logistic Regression Coefficients")
plt.axvline(0, color="black")
plt.tight_layout()
plt.show()
