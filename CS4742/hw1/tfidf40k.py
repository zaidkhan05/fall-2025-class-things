import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import argparse
import time

datafile = 'data.csv'
trainlines = 40000

print(f"Loading data from: {datafile}")
print(f"Using {trainlines} lines for training\n")

# Load the dataset
df = pd.read_csv(
    datafile,
    sep=",",
    header=0,
    names=["label", "text"],
    on_bad_lines="skip",
    encoding_errors="ignore",
    engine="python"
)

# Clean the data
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df["text"] = df["text"].astype(str).str.strip()
df = df.dropna(subset=["label", "text"])
df["label"] = df["label"].astype(int)

print(f"Total samples loaded: {len(df)}")

# Shuffle the rows randomly so training/testing split is random every time
df = df.sample(frac=1).reset_index(drop=True)


# Split into training and testing
train_df = df.iloc[:trainlines].copy()
test_df = df.iloc[trainlines:].copy()

print(f"Training samples: {len(train_df)}")
print(f"Testing samples:  {len(test_df)}\n")

X_train, y_train = train_df["text"], train_df["label"]
X_test, y_test = test_df["text"], test_df["label"]

# Create TF-IDF vectorizer
print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    max_features=5000
)
start_time = time.time()

# Transform the text into TF-IDF features
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}\n")

# Train the model
print("Training Logistic Regression model...")
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)
train_time = time.time() - start_time

# Make predictions
print("Making predictions...")
start_time = time.time()
y_pred = clf.predict(X_test_tfidf)
inference_time = time.time() - start_time

# Evaluate the model
print("\n" + "="*50)
print("TF-IDF Logistic Regression Results")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Training time: {train_time:.2f}s")
print(f"Inference time: {inference_time:.2f}s")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Show the most important words for positive/negative sentiment
print("\n" + "="*50)
print("Most Important Words")
print("="*50)

feature_names = tfidf.get_feature_names_out()
coefficients = clf.coef_[0]

top_pos_idx = np.argsort(coefficients)[-15:]
top_neg_idx = np.argsort(coefficients)[:15]

print("\nTop 15 Positive Words (predict positive reviews):")
for idx in reversed(top_pos_idx):
    print(f"  {feature_names[idx]:<20}  {coefficients[idx]:>7.3f}")

print("\nTop 15 Negative Words (predict negative reviews):")
for idx in top_neg_idx:
    print(f"  {feature_names[idx]:<20}  {coefficients[idx]:>7.3f}")

print("\n" + "="*50)
print("Done!")
print("="*50)
