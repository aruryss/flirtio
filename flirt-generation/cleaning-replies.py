import pandas as pd

# ----------------------
# Load the CSV with replies
# ----------------------
df = pd.read_csv("/Users/alibekabilmazhit/flirtio/flirt-generation/detection_train_with_replies_cleaned.csv")

# ----------------------
# Remove triple quotes from the 'reply' column
# ----------------------
df['reply'] = df['reply'].str.replace('"', '', regex=False)

# ----------------------
# Save the cleaned CSV
# ----------------------
df.to_csv("flirt-generation/detection_train_with_replies_cleaned.csv", index=False)
print("✓ Cleaned quotes from replies and saved to detection_train_with_replies.csv")
print("\nSample of cleaned data:")
print(df.head())