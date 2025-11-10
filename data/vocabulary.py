import pickle
from collections import Counter

def build_vocab(texts, min_freq=2, max_size=10000):
    all_tokens = []
    for text in texts:
        tokens = text.split()
        all_tokens.extend(tokens)
    
    word_counts = Counter(all_tokens)
    filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
    sorted_words = sorted(filtered_words, key=lambda x: word_counts[x], reverse=True)
    
    if max_size:
        sorted_words = sorted_words[:max_size]
    
    word_to_idx = {word: idx for idx, word in enumerate(sorted_words)}
    print(f"Vocabulary size: {len(word_to_idx)}")
    return word_to_idx

def add_special_tokens(word_to_idx):
    special_tokens = ['<PAD>', '<START>', '<END>', '<OOV>']
    shifted = {word: idx + len(special_tokens) for word, idx in word_to_idx.items()}
    
    for i, token in enumerate(special_tokens):
        shifted[token] = i
    
    return shifted

def save_vocab(word_to_idx, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(word_to_idx, f)
    print(f"Vocabulary saved to {filepath}")

def load_vocab(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def create_idx_to_word(word_to_idx):
    return {idx: word for word, idx in word_to_idx.items()}