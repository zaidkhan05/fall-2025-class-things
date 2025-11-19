# ---- run_entity_torch.py ----
import torch
import numpy as np
import pandas as pd
from model import (
    EntityMLP, token_overlap, entity_overlap, cosine_tfidf, extract_entities
)

model = EntityMLP()
model.load_state_dict(torch.load("entity_overlap_model.pt"))
model.eval()

# Load random headline and body from news.csv
df = pd.read_csv("news.csv")
sample = df.sample(n=1, random_state=None).iloc[0]
h = str(sample["title"]) if "title" in df.columns else str(sample.iloc[0])
b = str(sample["body"]) if "body" in df.columns else str(sample.iloc[1])

print(f"Headline: {h}")
print(f"Body: {b[:200]}..." if len(b) > 200 else f"Body: {b}")
print()

features = np.array([[
    token_overlap(h, b),
    entity_overlap(h, b),
    cosine_tfidf(h, b),
    len(extract_entities(h)),
    len(extract_entities(b)),
]], dtype=np.float32)

x = torch.tensor(features)
pred = model(x).item()

print("\nScore:", pred)
print("Prediction:", "Not Misleading" if pred >= 0.5 else "Misleading")
