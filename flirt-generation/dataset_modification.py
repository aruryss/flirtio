import pandas as pd
from openai import OpenAI
import os
import time


# ----------------------
# 1. Initialize OpenAI client
# ----------------------
client = OpenAI()
# API key is automatically read from OPENAI_API_KEY environment variable

# ----------------------
# 2. Load your dataset
# ----------------------
df = pd.read_csv("/Users/alibekabilmazhit/flirtio/data/processed/detection_train.csv")
print(f"Loaded {len(df)} rows.")

# ----------------------
# 3. Helper function to generate reply
# ----------------------
def generate_reply(text, label, max_retries=5):
    """
    Generate a reply using OpenAI GPT.
    label = 1 -> flirty
    label = 0 -> non-flirty
    """
    
    instruction = (
        "Generate a short flirty reply (max 12 words, playful, non-explicit)." if label == 1 
        else "Generate a short non-flirty friendly reply (max 12 words)."
    )

    prompt = f"Input: \"{text}\"\nInstruction: {instruction}\nOutput only the reply."

    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=50
            )
            reply = completion.choices[0].message.content.strip()
            return reply
        except Exception as e:
            print(f"Error: {e}, retrying...")
            retries += 1
            time.sleep(2)
    return ""  # return empty string if all retries fail

# ----------------------
# 4. Generate replies for the dataset
# ----------------------
input_texts = []
labels = []
replies = []

for idx, row in df.iterrows():
    text = row['text']
    label = int(row['label'])
    reply = generate_reply(text, label)
    print(f"[{idx}] Input: {text}\nLabel: {label} → Reply: {reply}\n")
    
    input_texts.append(text)
    labels.append(label)
    replies.append(reply)
    
    # optional: sleep to avoid rate limits
    time.sleep(1)

# ----------------------
# 5. Save as CSV
# ----------------------
output_df = pd.DataFrame({
    "input": input_texts,
    "label": labels,
    "reply": replies
})

output_df.to_csv("flirt-generation/detection_train_with_replies.csv", index=False)
print("✓ Dataset saved to detection_train_with_replies.csv")