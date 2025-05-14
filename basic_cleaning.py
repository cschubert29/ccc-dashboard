import pandas as pd
import re

file_path = "C:/Users/jamie/OneDrive/Documents/CCC Project/ccc-phase3-public.csv"
# Load dataset
df = pd.read_csv(file_path,
    encoding='latin1',
    low_memory=False
)

# Normalize text
def normalize_text(text):
    text = text.lower().strip()
    replacements = {
        r'\bpreisdent\b': 'president',
        r'\bdonal\b': 'donald',
        r'administrationas?': 'administration',
        r'\badminstration\b': 'administration',
        r'\badminstrative\b': 'administrative',
        r'\bpoliciesa\b': 'policies',
        r'\bpowr\b': 'power',
        r'\btrumpa\b': 'trump',
        r'\bmuskas?\b': 'musk',
        r'\binaguration\b': 'inauguration',
        r'\bmass deporation\b': 'mass deportation',
        r'\bexecutive order aensuring accountability for all agenciesa\b': 'executive order ensuring accountability for all agencies',
        r'\bdepartment of education\b': 'department of education',
        r'\bdei programs?\b': 'dei',
        r'\bin support of gov\b': 'in support of governor',
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text

# Apply normalization
df['claims_verbatim'] = df['claims_verbatim'].astype(str).apply(normalize_text)

# Define stance classifier
def classify_trump_stance(text):
    if "trump" not in text:
        return "Unclear"
    
    anti_patterns = [
        r"against .*trump", 
        r"oppose.*trump", 
        r"stop.*trump", 
        r"reject.*trump",
        r"refuse.*trump", 
        r"resist.*trump", 
        r"remove.*trump", 
        r"impeach.*trump",
        r'\bfuck trump\b',
        r'\bdump trump\b',
        r'\btrump sucks\b',
        r'\bno trump\b',
        r'\bstop trump\b',
        r'\bimpeach trump\b',
        r'\btrump is a (felon|dictator|loser|rapist)\b',
        r'\bhate.*trump\b',
        r'\bresist.*trump\b',
        r'\btrump.*got to go\b',
        r'\btrump +musk\b',
        r'\btrump =.*\b',  # Aiming here for "trump = felon"?
        r'\btrump +[a-z ]*devil'
    ]
    pro_patterns = [
        r"support.*trump", 
        r"celebrate.*trump", 
        r"for .*trump", 
        r"in favor of.*trump",
        r"stand with.*trump", 
        r"in solidarity with.*trump", 
        r'\btrump 2024\b',
        r'\bmake america great again\b',
        r'\btrump won\b',
        r'\btrump flag\b',
        r'\bwomen for trump\b',
        r'\bstand with trump\b',
        r'\bin support of .*trump\b',
        r'\bcelebrate.*trump\b'
    ]
    
    for pattern in anti_patterns:
        if re.search(pattern, text):
            return "Anti-Trump"
    for pattern in pro_patterns:
        if re.search(pattern, text):
            return "Pro-Trump"
    
    if re.search(r"anti-trump", text):
        return "Leaning Pro-Trump"
    if re.search(r"pro-trump", text):
        return "Leaning Anti-Trump"
    
    return "Unclear"

# Classify
df['trump_stance'] = df['claims_verbatim'].apply(classify_trump_stance)

# Filter to claims mentioning Trump
df_trump = df[df['claims_verbatim'].str.contains("trump", case=False, na=False)].copy()

# Export to CSV
# df_trump[['claims_verbatim', 'trump_stance']].drop_duplicates().to_csv("trump_classified_output.csv", index=False)

# Summary stats
print("\nStance Breakdown:")
print(df_trump['trump_stance'].value_counts())

# Display sample
print("\nSample Output:\n")
for _, row in df_trump[['claims_verbatim', 'trump_stance']].drop_duplicates().sample(
        min(100, len(df_trump)), random_state=1).iterrows():
    print(f"{row['trump_stance']:>15} | {row['claims_verbatim']}")
