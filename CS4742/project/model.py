# ---- entity_overlap_torch.py ----
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

nlp = spacy.load("en_core_web_sm")


# -------------------------------
# TEXT PREPROCESSING
# -------------------------------
def preprocess(text):
    text = str(text) if text else ""
    doc = nlp(text.lower())
    return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]


def extract_entities(text):
    text = str(text) if text else ""
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents]


def token_overlap(headline, body):
    h = set(preprocess(headline))
    b = set(preprocess(body))
    return len(h & b) / len(h) if len(h) > 0 else 0.0


def entity_overlap(headline, body):
    h = set(extract_entities(headline))
    b = set(extract_entities(body))
    return len(h & b) / len(h) if len(h) > 0 else 0.0


def cosine_tfidf(h, b):
    h = str(h) if h else ""
    b = str(b) if b else ""
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform([h, b])
    return (tfidf[0] @ tfidf[1].T).toarray()[0][0]


# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(df):
    X = []
    for _, row in df.iterrows():
        h, b = row["title"], row["body"]

        X.append([
            token_overlap(h, b),
            entity_overlap(h, b),
            cosine_tfidf(h, b),
            len(extract_entities(h)),
            len(extract_entities(b)),
        ])
    return np.array(X, dtype=np.float32)


# -------------------------------
# SIMPLE FEED-FORWARD MODEL
# -------------------------------
class EntityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


# -------------------------------
# TRAINING PIPELINE
# -------------------------------
if __name__ == "__main__":
    df = pd.read_csv("news.csv")

    # You must supply labels
    # 1 = not misleading
    # 0 = misleading
    # TEMP: random labels until you annotate
    df["label"] = np.random.randint(0, 2, df.shape[0])

    # Use only a portion of the data for faster processing
    SAMPLE_SIZE = 3000  # Adjust this number as needed
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"Using {SAMPLE_SIZE} samples from dataset")
    
    print("Extracting features...")
    X = extract_features(df)
    y = df["label"].values.astype(np.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EntityMLP().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0007)

    X_train_t = torch.tensor(X_train).to(device)
    y_train_t = torch.tensor(y_train).to(device)
    X_test_t = torch.tensor(X_test).to(device)
    y_test_t = torch.tensor(y_test).to(device)

    EPOCHS = 25

    for epoch in range(EPOCHS):
        model.train()

        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    preds = (model(X_test_t).detach().cpu().numpy() >= 0.5).astype(int)
    acc = (preds == y_test).mean()

    print(f"\nTest Accuracy: {acc:.4f}")

    # Save model
    torch.save(model.state_dict(), "entity_overlap_model.pt")
    print("Model saved → entity_overlap_model.pt")
