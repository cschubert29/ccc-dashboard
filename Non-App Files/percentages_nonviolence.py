import pandas as pd
import io

# Load the CSV into a DataFrame, handling potential delimiter issues and skipping blank lines
# Use a more robust delimiter detection (if needed) or specify the delimiter explicitly
# Example: df = pd.read_csv("ccc_anti_trump.csv", sep=',', skipinitialspace=True, skip_blank_lines=True)
with open("ccc_anti_trump.csv", 'r', encoding='utf-8') as f:
    data = f.read()

# Replace problematic text entries with NaN before reading into DataFrame
replacements = {
    'unspecified': '0',
    'graffiti': '0',
    'vandalism': '0',
    # Add other text descriptions as needed, mapping them to appropriate numeric values or NaN
}

for old, new in replacements.items():
    data = data.replace(old, new)

df = pd.read_csv(io.StringIO(data), skipinitialspace=True, skip_blank_lines=True)

# Force numeric fields, coerce errors to NaN
fields = [
    'participant_injuries', 'police_injuries',
    'arrests', 'arrests_any',
    'property_damage', 'property_damage_any',
    'participant_casualties_any', 'police_casualties_any',
    'participant_deaths', 'police_deaths'
]

for field in fields:
    df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)

# Calculate no injuries
df['no_participant_injuries'] = (df['participant_injuries'] == 0)
df['no_police_injuries'] = (df['police_injuries'] == 0)

# Calculate no arrests
df['no_arrests'] = (df['arrests'] == 0) & (df['arrests_any'] == 0)

# Calculate no property damage
df['no_property_damage'] = (df['property_damage'] == 0) & (df['property_damage_any'] == 0)

# Calculate no casualties
df['no_participant_casualties'] = (df['participant_casualties_any'] == 0)
df['no_police_casualties'] = (df['police_casualties_any'] == 0)

# Calculate no deaths
df['no_participant_deaths'] = (df['participant_deaths'] == 0)
df['no_police_deaths'] = (df['police_deaths'] == 0)

# Get percentages
summary = {
    'No Participant Injuries %': 100 * df['no_participant_injuries'].mean(),
    'No Police Injuries %': 100 * df['no_police_injuries'].mean(),
    'No Arrests %': 100 * df['no_arrests'].mean(),
    'No Property Damage %': 100 * df['no_property_damage'].mean(),
    'No Participant Casualties %': 100 * df['no_participant_casualties'].mean(),
    'No Police Casualties %': 100 * df['no_police_casualties'].mean(),
    'No Participant Deaths %': 100 * df['no_participant_deaths'].mean(),
    'No Police Deaths %': 100 * df['no_police_deaths'].mean(),
}

# Show summary
for key, value in summary.items():
    print(f"{key}: {value:.2f}%")

# Output rows with incidents
incident_rows = df[
    (df['participant_injuries'] > 0) |
    (df['police_injuries'] > 0) |
    (df['arrests'] > 0) |
    (df['arrests_any'] > 0) |
    (df['property_damage'] > 0) |
    (df['property_damage_any'] > 0) |
    (df['participant_casualties_any'] > 0) |
    (df['police_casualties_any'] > 0) |
    (df['participant_deaths'] > 0) |
    (df['police_deaths'] > 0)
]

# Save the entire incident rows DataFrame to CSV
incident_rows.to_csv('incident_data.csv', index=False)

print("\nIncident data saved to incident_data.csv")
