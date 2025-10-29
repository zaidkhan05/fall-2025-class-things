

````markdown
# 🧠 CS4742: Natural Language Processing — Assignment 1 (Fall 2025)
### Kennesaw State University  
**Instructor:** M. Alexiou  
**Due Date:** October 31, 2025  
**Topic:** Logistic Regression for Sentiment Analysis  

---

## 📘 Overview

In this assignment, you will **implement Logistic Regression** in **Python** for **binary sentiment classification** (positive or negative).  
You’ll train and test your model using the **Amazon Product Reviews dataset** and explore how preprocessing, feature extraction, and training size affect performance.

The purpose of this project is to **understand logistic regression**, **feature representation**, and **evaluation metrics** in text-based classification.

---

## 📂 Dataset

Use the dataset from the following link:

🔗 [Amazon Reviews Dataset](https://github.com/MuhammedBuyukkinaci/TensorFlow-Sentiment-Analysis-on-Amazon-Reviews-Data/blob/master/dataset/)

This dataset contains product reviews labeled as **positive (1)** or **negative (0)**.

> 💡 **Tip:** The dataset is large — start with 40,000 reviews and gradually scale up to 80,000 to observe accuracy trends.

---

## ⚙️ Implementation Steps

### 1. Setup and Dependencies

Create a virtual environment and install necessary libraries:

```bash
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)
pip install pandas numpy scikit-learn matplotlib nltk
````

If needed for text preprocessing:

```bash
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
```

---

### 2. Folder Structure

Organize your files as follows:

```
Assignment1_ZaidKhan/
│
├── README.md
├── logistic_regression_bow.py
├── logistic_regression_tfidf.py
├── utils/
│   ├── preprocess.py
│   ├── vectorize.py
│   └── evaluate.py
├── report.pdf
└── results/
    ├── confusion_matrix_bow.png
    ├── confusion_matrix_tfidf.png
    └── accuracy_comparison.csv
```

This helps maintain clarity and makes grading easier.

---

### 3. Data Preprocessing

Before training, clean and normalize your text data.

Typical preprocessing steps:

* Convert text to lowercase
* Remove punctuation and stopwords
* Tokenize text (split into words)
* Optionally apply stemming or lemmatization

**Example (`utils/preprocess.py`):**

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)
```

---

### 4. Data Representation (Feature Extraction)

You can’t feed raw text into logistic regression — it must be converted into numerical form.

#### Option 1: Bag of Words (BoW)

Represents text as counts of word occurrences.

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus).toarray()
```

#### Option 2: TF-IDF

Gives higher weight to words that are unique to certain documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus).toarray()
```

> 💡 Compare both BoW and TF-IDF to see which yields better accuracy and efficiency.

---

### 5. Splitting the Data

Divide the data into **training** and **testing** sets:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 6. Logistic Regression Model

Train a logistic regression model on your vectorized data.

```python
from sklearn.linear_model import LogisticRegression
import time

start = time.time()
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
training_time = time.time() - start

print(f"Training time: {training_time:.2f} seconds")
```

---

### 7. Evaluation and Metrics

Evaluate the model using **accuracy** and a **confusion matrix**.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

#### Visualize the Confusion Matrix

```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (TF-IDF)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

Save plots in the `results/` folder.

---

### 8. Efficiency Measurement

Measure:

* **Training Time**
* **Prediction (Inference) Time**

Use Python’s `time` module before and after model fitting/predicting.

---

### 9. Experiment with Dataset Sizes

Test your model with:

* 40,000 samples
* 60,000 samples
* 80,000 samples

Record how accuracy and training time change.

---

### 10. Results and Discussion

In your **report.pdf**, include:

* Model parameters and choices (vectorization type, `max_iter`, etc.)
* Dataset size comparisons
* Accuracy and timing results
* Confusion matrices for both implementations
* Discussion of why one approach performed better (based on sparsity, feature scaling, etc.)

---

## 🧾 Report Format

Your PDF report should include:

| Section             | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **1. Introduction** | Purpose of the experiment and dataset overview               |
| **2. Methodology**  | Preprocessing, feature extraction, logistic regression setup |
| **3. Experiments**  | Parameters, training sizes, and model configurations         |
| **4. Results**      | Tables and plots (accuracy, confusion matrices, efficiency)  |
| **5. Discussion**   | Analysis and interpretation of results                       |
| **6. Conclusion**   | Summary of findings and potential improvements               |

---

## 📦 Submission Instructions

Submit a **single ZIP archive** named:

```
CS4742_Assignment1_TeamName.zip
```

Containing:

* Source code (`.py` files)
* `report.pdf`
* `README.md` with:

  * Team member names
  * KSU email addresses
  * Instructions to run the code

Upload to **D2L → Assignments → HW1**

### ⚠️ Late Submission Penalty

* **–10% per day** after the deadline.

---

## ✅ Grading Criteria

| Component                                  | Weight |
| ------------------------------------------ | ------ |
| Correct Logistic Regression Implementation | 30%    |
| Preprocessing & Feature Representation     | 20%    |
| Comparison of Two Implementations          | 20%    |
| Report Quality & Discussion                | 20%    |
| Code Organization & Readability            | 10%    |

---

## 💡 Bonus Tips

* Use `random_state` for reproducibility.
* Try `solver='saga'` or `solver='liblinear'` for large datasets.
* Limit features (`max_features=5000`) to balance performance and speed.
* Save your models using `joblib` if you plan multiple runs.

---

**End of Document**

```

---

Would you like me to also generate a **starter code template** (`logistic_regression_bow.py` and `logistic_regression_tfidf.py`) that matches this Markdown guide? It’ll include all the structure, imports, and comments so you can run and record results immediately.
```
