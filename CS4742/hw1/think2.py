# ---- /models/logreg_tfidf_split.py ----
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import joblib

# 1) Load single dataset safely
df = pd.read_csv(
    "data.csv",
    sep=",",
    header=0,
    names=["label", "text"],
    on_bad_lines="skip",
    encoding_errors="ignore",
    engine="python"
)

# 2) Clean and prepare labels/text
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df["text"]  = df["text"].astype(str).str.strip()
df = df.dropna(subset=["label", "text"])
df["label"] = df["label"].astype(int)

print(f"Total samples loaded: {len(df)}")

# 3) Split into training (first 40k) and testing (remaining)
train_df = df.iloc[:40000].copy()
test_df  = df.iloc[40000:].copy()

print(f"Training samples: {len(train_df)}")
print(f"Testing samples:  {len(test_df)}")

X_train, y_train = train_df["text"], train_df["label"]
X_test, y_test   = test_df["text"],  test_df["label"]

# 4) TF-IDF Vectorization
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9,
    max_features=10000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# 5) Train logistic regression
clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
clf.fit(X_train_tfidf, y_train)

# 6) Evaluate
y_pred = clf.predict(X_test_tfidf)
print("\n===== TF-IDF Logistic Regression Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred, digits=3))

# 7) Interpret model weights — top positive vs. negative words
feature_names = tfidf.get_feature_names_out()
coefficients = clf.coef_[0]

top_pos_idx = np.argsort(coefficients)[-20:]
top_neg_idx = np.argsort(coefficients)[:20]

print("\nTop Positive Words:")
for idx in reversed(top_pos_idx):
    print(f"{feature_names[idx]:<20}  {coefficients[idx]:.3f}")

print("\nTop Negative Words:")
for idx in top_neg_idx:
    print(f"{feature_names[idx]:<20}  {coefficients[idx]:.3f}")

# 8) Save model + vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(clf, "logreg_tfidf_model.pkl")
print("\nSaved: tfidf_vectorizer.pkl and logreg_tfidf_model.pkl")
