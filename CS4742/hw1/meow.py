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

#####################################
# ██████╗  █████╗ ████████╗ █████╗  #
# ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗ #
# ██║  ██║███████║   ██║   ███████║ #
# ██║  ██║██╔══██║   ██║   ██╔══██║ #
# ██████╔╝██║  ██║   ██║   ██║  ██║ #
# ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ #
#####################################
                                 
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
trainCount = 80000
#use 40k lines for training, rest for testing
X_train, y_train = x[:trainCount], y[:trainCount]
X_test, y_test = x[trainCount:], y[trainCount:]

train_df = pd.DataFrame({'text': X_train, 'label': y_train})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")



#######################################################
#  _________  ________  ________  ___  ________       #
# |\___   ___\\   __  \|\   __  \|\  \|\   ___  \     #
# \|___ \  \_\ \  \|\  \ \  \|\  \ \  \ \  \\ \  \    #
#      \ \  \ \ \   _  _\ \   __  \ \  \ \  \\ \  \   #
#       \ \  \ \ \  \\  \\ \  \ \  \ \  \ \  \\ \  \  #
#        \ \__\ \ \__\\ _\\ \__\ \__\ \__\ \__\\ \__\ #
#         \|__|  \|__|\|__|\|__|\|__|\|__|\|__| \|__| #
#######################################################

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

###########################################################################
# BBBB    A    GGG         OOO  FFFFF       W   W  OOO  RRRR  DDDD   SSSS #
# B   B  A A  G           O   O F           W   W O   O R   R D   D S     #
# BBBB  AAAAA G GG        O   O FFFF        W W W O   O RRRR  D   D  SSS  #
# B   B A   A G   G       O   O F           W W W O   O R  R  D   D     S #
# BBBB  A   A  GGG         OOO  F            W W   OOO  R   R DDDD  SSSS  #
###########################################################################

# METHOD 1: Bag-of-Words (count how many times each word appears)
bow_vectorizer = CountVectorizer(stop_words='english', max_features=5000)
results_bow = train_and_evaluate(bow_vectorizer, "Bag-of-Words Model")

###############################################################
#  mmmmmmmm  mmmmmmmm             mmmmmm   mmmmm     mmmmmmmm #
#  """##"""  ##""""""             ""##""   ##"""##   ##"""""" #
#     ##     ##                     ##     ##    ##  ##       #
#     ##     #######                ##     ##    ##  #######  #
#     ##     ##         #####       ##     ##    ##  ##       #
#     ##     ##                   mm##mm   ##mmm##   ##       #
#     ""     ""                   """"""   """""     ""       #
###############################################################
                                                            
# METHOD 2: TF-IDF (analyze word importance across all reviews)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
results_tfidf = train_and_evaluate(tfidf_vectorizer, "TF-IDF Model")

################################################################
#                               mm                             #
#                               MM                             #
#  ,p6"bo `7MM  `7MM  ,pP"Ybd mmMMmm ,pW"Wq.`7MMpMMMb.pMMMb.   #
# 6M'  OO   MM    MM  8I   `"   MM  6W'   `Wb MM    MM    MM   #
# 8M        MM    MM  `YMMMa.   MM  8M     M8 MM    MM    MM   #
# YM.    ,  MM    MM  L.   I8   MM  YA.   ,A9 MM    MM    MM   #
#  YMbmd'   `Mbod"YML.M9mmmP'   `Mbmo`Ybmd9'.JMML  JMML  JMML. #
################################################################

def extract_custom_features(df):
    """This is like being a detective - we look for special clues in each review!"""
    # good words (review is probably positive)
    # positive_words = ['good', 'great', 'awesome', 'excellent', 'love', 'fantastic', 'amazing', 'wonderful', 
    #                   'best', 'perfect', 'brilliant', 'outstanding', 'superb', 'enjoyed', 'recommend']
    positive_words = ['great', 'excellent', 'best', 'love', 'awesome', 'perfect', 'amazing', 'wonderful', 'fantastic',
                      'loves', 'highly', 'favorite', 'easy', 'loved', 'good', 'pleased', 'works', 'fun', 'highly recommend', 'enjoyed']

    # bad words (=review is probably negative)
    # negative_words = ['bad', 'worst', 'awful', 'terrible', 'hate', 'poor', 'disappoint', 'boring', 
    #                   'waste', 'horrible', 'dull', 'stupid', 'annoying', 'pointless', 'fails']
    negative_words = ['disappointing', 'worst', 'disappointed', 'poor', 'boring', 'waste', 'terrible',
                      'disappointment', 'horrible', 'poorly', 'awful', 'useless', 'return', 'unfortunately', 'bad', 'returned', 'sorry', 'worse', 'junk', 'mediocre']
    
    # # negations ("not good" = bad!)
    # negation_words = ['not', 'never', 'no', "n't", 'nothing', 'nowhere', 'neither', 'nobody', 'none']

    feats = []
    # Look at each review one by one
    for text in df['text']:
        original_text = str(text)
        text_lower = original_text.lower()  # Make everything lowercase
        words = re.findall(r'\w+', text_lower)  # Split into individual words
        word_count = len(words) if words else 1  # How many words total?

        #Basic text features
        avg_word_len = np.mean([len(w) for w in words]) if words else 0  # Long words = fancy?
        exclam = text_lower.count('!')  # Lots of ! = excited?
        question = text_lower.count('?')  # Lots of ? = confused?

        # Sentiment features (good vs bad words)
        pos_count = sum(w in positive_words for w in words)
        neg_count = sum(w in negative_words for w in words)
        pos_ratio = pos_count / word_count  # What % of words are good
        neg_ratio = neg_count / word_count  # What % of words are bad
        sentiment_diff = pos_count - neg_count  # More good or bad words
        
        # #Negation features (words that flip meaning)
        # negation_count = sum(w in negation_words for w in words)
        # negation_ratio = negation_count / word_count  # What % are negation words
        
        #Capitalization (YELLING?)
        caps_count = sum(1 for word in original_text.split() if word.isupper() and len(word) > 1)
        
        #Structure features
        sentence_count = len(re.split(r'[.!?]+', text_lower))  # How many sentences
        has_digits = int(bool(re.search(r'\d', text_lower)))  # Does it have numbers
        
        #First and last word (people often start/end with their true feeling)
        first_word_pos = int(words[0] in positive_words) if words else 0  # Start with happy word
        first_word_neg = int(words[0] in negative_words) if words else 0  # Start with sad word
        last_word_pos = int(words[-1] in positive_words) if words else 0  # End with happy word
        last_word_neg = int(words[-1] in negative_words) if words else 0  # End with sad word
        
        #Multiple punctuation (like !!! or ???) = REALLY excited/confused
        multi_punct = len(re.findall(r'[!?]{2,}', text_lower))

        #feature list for this review
        feats.append([
            avg_word_len, exclam, question, pos_count, neg_count, 
            pos_ratio, neg_ratio, sentiment_diff,
            caps_count, sentence_count, has_digits, first_word_pos, first_word_neg,
            last_word_pos, last_word_neg, multi_punct, word_count
        ])

    features_df = pd.DataFrame(feats, columns=[
        'avg_word_length', 'exclamation_count', 'question_count', 'positive_word_count',
        'negative_word_count', 'positive_ratio', 'negative_ratio', 'sentiment_diff',
        'caps_count', 'sentence_count', 'has_digits', 'first_word_pos', 'first_word_neg',
        'last_word_pos', 'last_word_neg', 'multi_punct', 'text_length'
    ])
    return features_df

#Run custom feature model
print("\n===== Custom Feature Model =====")
# gather features from reviews
X_train_custom = extract_custom_features(train_df)
X_test_custom = extract_custom_features(test_df)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_custom)
X_test_scaled = scaler.transform(X_test_custom)

#logistic regression model
model_custom = LogisticRegression(max_iter=1000, solver='lbfgs')

#train model on custom features
start_train = time.time()
model_custom.fit(X_train_scaled, y_train)
train_time_custom = time.time() - start_train

#test model
start_infer = time.time()
y_pred_custom = model_custom.predict(X_test_scaled)
infer_time_custom = time.time() - start_infer

#measure performance
acc_custom = accuracy_score(y_test, y_pred_custom)
cm_custom = confusion_matrix(y_test, y_pred_custom)

print(f"Accuracy: {acc_custom:.4f}")
print(f"Training time: {train_time_custom:.2f}s | Inference time: {infer_time_custom:.2f}s")
print("Confusion Matrix:\n", cm_custom)
print("\nClassification Report:\n", classification_report(y_test, y_pred_custom, digits=4))

###########################################################################
# (   __ \    / ___/   / ____\  ) )  ( (  (_   _)     (___  ___)  / ____\ #
#  ) (__) )  ( (__    ( (___   ( (    ) )   | |           ) )    ( (___   #
# (    __/    ) __)    \___ \   ) )  ( (    | |          ( (      \___ \  #
#  ) \ \  _  ( (           ) ) ( (    ) )   | |   __      ) )         ) ) #
# ( ( \ \_))  \ \___   ___/ /   ) \__/ (  __| |___) )    ( (      ___/ /  #
#  )_) \__/    \____\ /____/    \______/  \________/     /__\    /____/   #
###########################################################################

# Final summary of all methods
print("\n===== SUMMARY =====")
print(f"Bag-of-Words Accuracy: {results_bow[0]:.4f}")
print(f"TF-IDF Accuracy:       {results_tfidf[0]:.4f}")
print(f"Custom Features Accuracy: {acc_custom:.4f}")

print("\nResults saved in 'results_log.txt'.")

# Close our notebook file properly (like closing a book when done)
log_file.close()
