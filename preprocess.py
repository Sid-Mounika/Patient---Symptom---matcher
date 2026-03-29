import pandas as pd
import re

# Load original dataset
df = pd.read_csv("symptoms_df.csv")  # make sure filename matches yours

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = text.replace("_", " ")      # skin_rash → skin rash
    text = re.sub(r"\s+", " ", text)   # remove extra spaces
    text = text.strip().lower()        # normalize
    
    return text

def combine_symptoms(row):
    symptoms = [
        clean_text(row["Symptom_1"]),
        clean_text(row["Symptom_2"]),
        clean_text(row["Symptom_3"]),
        clean_text(row["Symptom_4"])
    ]
    
    # Remove empty values
    symptoms = [s for s in symptoms if s != ""]
    
    # Join into single sentence (for embeddings)
    return ", ".join(symptoms)

# Create combined column
df["combined_symptoms"] = df.apply(combine_symptoms, axis=1)

# Remove duplicate symptom rows (important for FAISS)
df = df.drop_duplicates(subset=["combined_symptoms"]).reset_index(drop=True)

# 🔥 THIS LINE SAVES THE CLEAN DATASET
output_file = "cleaned_symptom_dataset.csv"
df.to_csv(output_file, index=False)

# Show full text in console (no ...)
pd.set_option('display.max_colwidth', None)

print("\nCleaned Dataset Preview:\n")
print(df[["Disease", "combined_symptoms"]].head())
print(f"\n✅ Cleaned dataset saved as: {output_file}")
print(f"📁 Location: same folder as preprocess.py")
print(f"📊 Total records: {len(df)}")
