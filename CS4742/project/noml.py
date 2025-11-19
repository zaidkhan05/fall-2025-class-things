# ---- entity_overlap_model.py ----
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

# load spaCy model
nlp = spacy.load("en_core_web_sm")


# ---------------------------------
# TEXT PREPROCESSING + LEMMATIZATION
# ---------------------------------
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return tokens


# ---------------------------------
# EXTRACT NAMED ENTITIES
# ---------------------------------
def extract_entities(text):
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents]


# ---------------------------------
# ENTITY + KEYWORD OVERLAP SCORING
# ---------------------------------
def compute_overlap_score(headline_tokens, body_tokens):
    if not headline_tokens or not body_tokens:
        return 0.0

    set_headline = set(headline_tokens)
    set_body = set(body_tokens)

    overlap = set_headline.intersection(set_body)

    score = len(overlap) / len(set_headline)
    return score, overlap


# ---------------------------------
# COMBINE ENTITY + TOKEN OVERLAP
# ---------------------------------
def combined_score(headline, body):
    headline_tokens = preprocess(headline)
    body_tokens = preprocess(body)

    headline_entities = extract_entities(headline)
    body_entities = extract_entities(body)

    token_score, token_overlap = compute_overlap_score(
        headline_tokens, body_tokens
    )

    entity_score, entity_overlap = compute_overlap_score(
        headline_entities, body_entities
    )

    # weighted average
    final_score = (0.7 * token_score) + (0.3 * entity_score)

    return {
        "token_score": round(token_score, 4),
        "entity_score": round(entity_score, 4),
        "final_score": round(final_score, 4),
        "token_overlap": list(token_overlap),
        "entity_overlap": list(entity_overlap),
    }


# ---------------------------------
# SIMPLE THRESHOLD CLASSIFIER
# ---------------------------------
def classify(score):
    # Adjust threshold during tuning
    THRESHOLD = 0.25
    return "Not Misleading" if score >= THRESHOLD else "Misleading"


# ---------------------------------
# MAIN TEST FUNCTION
# ---------------------------------
if __name__ == "__main__":
    df = pd.read_csv("news.csv")
    sample = df.sample(n=1, random_state=None).iloc[0]
    h = str(sample["title"]) if "title" in df.columns else str(sample.iloc[0])
    b = str(sample["body"]) if "body" in df.columns else str(sample.iloc[1])
    print(f"Headline: {h}")
    print(f"Body: {b[:200]}..." if len(b) > 200 else f"Body: {b}")
    print()
    headline = h
    body = b

    results = combined_score(headline, body)

    print("\n=== ENTITY/TOKEN OVERLAP MODEL OUTPUT ===")
    print(f"Token Overlap Score: {results['token_score']}")
    print(f"Entity Overlap Score: {results['entity_score']}")
    print(f"Final Score: {results['final_score']}")
    print(f"Token Overlap: {results['token_overlap']}")
    print(f"Entity Overlap: {results['entity_overlap']}")

    label = classify(results["final_score"])
    print(f"\nPrediction: {label}")
