import re
import numpy as np

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def tokenize_text(text):
    return text.split()

def create_sequences(texts, word_to_idx):
    oov_idx = word_to_idx.get('<OOV>', 3)
    sequences = []
    for text in texts:
        tokens = tokenize_text(text)
        seq = [word_to_idx.get(token, oov_idx) for token in tokens]
        sequences.append(seq)
    return sequences

def pad_sequences(sequences, maxlen, padding='post', value=0):
    padded = np.full((len(sequences), maxlen), value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        if padding == 'post':
            padded[i, :min(len(seq), maxlen)] = seq[:maxlen]
        else:
            padded[i, max(0, maxlen-len(seq)):] = seq[-maxlen:]
    return padded