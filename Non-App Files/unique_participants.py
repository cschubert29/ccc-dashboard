import pandas as pd
import re

# Load CSV file
file_path = "C:/Users/jamie/OneDrive/Documents/CCC Project/ccc-phase3-public.csv"
df = pd.read_csv(file_path, encoding='latin1', low_memory=False)

# Ensure the column is string type and handle missing
df['participants'] = df['participants'].astype(str).fillna('')

# Split on both commas and semicolons using regex, then explode and strip whitespace
participants = df['participants'].dropna().apply(lambda x: re.split(r'[;,]', x))
participants = participants.explode().str.strip()

# Drop blanks and get sorted unique values
distinct_participants = sorted(participants[participants != ''].unique())

# Print or save
for p in distinct_participants:
    print(p)

# Save to file
with open("distinct_participants.txt", "w", encoding="utf-8") as f:
    for p in distinct_participants:
        f.write(p + "\n")
