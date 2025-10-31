import pandas as pd
import numpy as np
import time
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

##################################################################
#                    ,----,                               ,----, #
#                  ,/   .`|                             ,/   .`| #
#   .--.--.      ,`   .'  : ,---,       ,-.----.      ,`   .'  : #
#  /  /    '.  ;    ;     /'  .' \      \    /  \   ;    ;     / #
# |  :  /`. /.'___,/    ,'/  ;    '.    ;   :    \.'___,/    ,'  #
# ;  |  |--` |    :     |:  :       \   |   | .\ :|    :     |   #
# |  :  ;_   ;    |.';  ;:  |   /\   \  .   : |: |;    |.';  ;   #
#  \  \    `.`----'  |  ||  :  ' ;.   : |   |  \ :`----'  |  |   #
#   `----.   \   '   :  ;|  |  ;/  \   \|   : .  /    '   :  ;   #
#   __ \  \  |   |   |  ''  :  | \  \ ,';   | |  \    |   |  '   #
#  /  /`--'  /   '   :  ||  |  '  '--'  |   | ;\  \   '   :  |   #
# '--'.     /    ;   |.' |  :  :        :   ' | \.'   ;   |.'    #
#   `--'---'     '---'   |  | ,'        :   : :-'     '---'      #
#                        `--''          |   |.'                  #
#                                       `---'                    #
##################################################################

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except ValueError:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except ValueError:
                pass

log_file = open("results_log.txt", "w", encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)


#dataset
DATAPATH = "data.csv"

print("Loading datasets...")

df = pd.read_csv(DATAPATH)

if not {'label', 'text'}.issubset(df.columns):
    raise ValueError(f"{DATAPATH} must contain 'label' and 'text' columns.")


print(f"Total samples: {len(df)}")

#shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate text from labels
x, y = df['text'], df['label']
trainCount = 40000
#use 40k lines for training, rest for testing
X_train, y_train = x[:trainCount], y[:trainCount]
X_test, y_test = x[trainCount:], y[trainCount:]

train_df = pd.DataFrame({'text': X_train, 'label': y_train})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")




def train_and_evaluate(vectorizer, model_name="Model"):
    print(f"\n===== {model_name} =====")

    # Turn words into numbers and apply to both train and test sets
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    

    # training
    start_train = time.time()
    model.fit(X_train_vec, y_train)
    train_time = time.time() - start_train

    # Test the model
    start_infer = time.time()
    y_pred = model.predict(X_test_vec)
    infer_time = time.time() - start_infer

    # Measure performance
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")  
    print(f"Training time: {train_time:.2f}s | Inference time: {infer_time:.2f}s")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    return acc, cm, train_time, infer_time


                                                            
# METHOD 2: TF-IDF (analyze word importance across all reviews)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
# 4) TF-IDF Vectorization
# tfidf = TfidfVectorizer(
#     lowercase=True,
#     stop_words="english",
#     ngram_range=(1,2),
#     min_df=2,
#     max_df=0.9,
#     max_features=10000
# )
results_tfidf = train_and_evaluate(tfidf_vectorizer, "TF-IDF Model")
feature_names = tfidf_vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_pos_idx = np.argsort(coefficients)[-20:]
top_neg_idx = np.argsort(coefficients)[:20]

print("\nTop Positive Words:")
for idx in reversed(top_pos_idx):
    print(f"{feature_names[idx]:<20}  {coefficients[idx]:.3f}")

print("\nTop Negative Words:")
for idx in top_neg_idx:
    print(f"{feature_names[idx]:<20}  {coefficients[idx]:.3f}")



# Final summary of all methods
print("\n===== SUMMARY =====")
print(f"TF-IDF Accuracy:       {results_tfidf[0]:.4f}")

print("\nResults saved in 'results_log.txt'.")

# Close our notebook file properly (like closing a book when done)
log_file.close()
