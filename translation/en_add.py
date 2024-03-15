import pandas as pd

# Read the english_texts.csv file
english_data = pd.read_csv("english_texts.csv")

# Extract the 'transcript' column
english_text = english_data["transcript"]

# Check if en_hin_translated.csv exists (optional)
try:
  # Read the existing en_hin_translated.csv file (if it exists)
  existing_data = pd.read_csv("en_hin_translated.csv")
except FileNotFoundError:
  # Create a new empty DataFrame if the file doesn't exist
  existing_data = pd.DataFrame(columns=["en", "hi"])

# Add the english text as a new 'en' column
existing_data["en"] = english_text

# Save the updated data to en_hin_translated.csv
existing_data.to_csv("en_hin_translated.csv", index=False)

print("English text data copied to en_hin_translated.csv")
