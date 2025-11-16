# ---- run_entity_torch.py ----
import torch
import numpy as np
from entity_overlap_torch import (
    EntityMLP, token_overlap, entity_overlap, cosine_tfidf, extract_entities
)

model = EntityMLP()
model.load_state_dict(torch.load("entity_overlap_model.pt"))
model.eval()

h = input("Headline: ")
b = input("Body: ")

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
