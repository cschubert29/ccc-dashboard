import pandas as pd
import re
from transformers import pipeline
import torch
from tqdm import tqdm

# Set device (GPU if available)
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'GPU' if device == 0 else 'CPU'}")

# Load CSV
df = pd.read_csv(r"C:\Users\jamie\Downloads\ccc-phase3-public.csv", encoding="latin1")
if 'claims_verbatim' not in df.columns:
    raise ValueError("Expected column 'claims_verbatim' not found in CSV.")

# Normalize text
def normalize_text(text):
    text = str(text).lower().strip()
    replacements = {
        r'\bpreisdent\b': 'president',
        r'\bdonal\b': 'donald',
        r'\btrumpa?\b': 'trump',
        r'\badminstration\b': 'administration',
        r'\badminstrative\b': 'administrative',
        r'\bpoliciesa\b': 'policies',
        r'\bpowr\b': 'power',
        r'\bmuskas?\b': 'musk',
        r'\bdei programs?\b': 'dei',
        r'\bdepora?tions?\b': 'deportations',
        r'\bfacism\b': 'fascism',
        r'\binaguration\b': 'inauguration',
        r'â|\x92': "'",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text

df["claims_verbatim"] = df["claims_verbatim"].apply(normalize_text)

# Hugging Face pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# Labels for classification
labels = ["Pro-Trump", "Anti-Trump", "Unclear"]

# Apply classification
texts = df["claims_verbatim"].tolist()
batch_size = 8
results = []
for i in tqdm(range(0, len(texts), batch_size), desc="Classifying"):
    batch = texts[i:i+batch_size]
    batch_results = classifier(batch, labels)
    results.extend(batch_results)

def extract_result(result):
    label = result['labels'][0]
    scores = dict(zip(result['labels'], result['scores']))
    return {
        "label": label,
        "Pro-Trump Score": scores.get("Pro-Trump", 0),
        "Anti-Trump Score": scores.get("Anti-Trump", 0),
        "Unclear Score": scores.get("Unclear", 0)
    }

classified = [extract_result(r) for r in results]
df_classified = pd.concat([df, pd.DataFrame(classified)], axis=1)

# Print results
print("\nStance Breakdown:")
print(df_classified["label"].value_counts().rename_axis("stance"))

# Print sample output
print("\nSample Output:\n")
sample = df_classified[["claims_verbatim", "label", "Pro-Trump Score", "Anti-Trump Score", "Unclear Score"]].drop_duplicates()
print(sample.sample(min(10, len(sample)), random_state=1).to_string(index=False))

# Optional: Save results
df_classified.to_csv("trump_classified_output.csv", index=False)
