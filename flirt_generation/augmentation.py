import random
import nltk
from nltk.corpus import wordnet
import pandas as pd

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

def synonym_replacement(text, n=2):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    
    num_replaced = 0
    for word in random_word_list:
        synonyms = get_synonyms(word)
        if len(synonyms) > 0:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if w == word else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    
    return ' '.join(new_words)

def augment_data(df, target_size=None):
    """Augment a dataframe with synonym replacement"""
    if target_size is None:
        target_size = len(df) * 2
    
    augmented = []
    augmented.extend(df.to_dict('records'))
    
    for _, row in df.iterrows():
        if len(augmented) >= target_size:
            break
        # Augment by replacing synonyms in the text
        augmented.append({
            'input': synonym_replacement(row['input']),
            'reply': synonym_replacement(row['reply']),
            'label': row['label']
        })
    
    return pd.DataFrame(augmented[:target_size])

train_df = pd.read_csv('/Users/alibekabilmazhit/flirtio/flirt-generation/detection_train_with_replies_cleaned.csv')
# val_df = pd.read_csv('data/processed/detection_val.csv')
# test_df = pd.read_csv('data/processed/detection_test.csv')

print(f"Original sizes - Train: {len(train_df)}, ")

augmented_train_df = augment_data(train_df, target_size=len(train_df) * 2)

print(f"Augmented train size: {len(augmented_train_df)}")

augmented_train_df.to_csv('flirt-generation/augmented-data/detection_train_augmented_with_replies.csv', index=False)
# val_df.to_csv('flirt-generation/augmented-data/detection_val_augmented.csv', index=False)
# test_df.to_csv('flirt-generation/augmented-data/detection_test_augmented.csv', index=False)

print("Augmented files saved:")
print("  - detection_train_augmented.csv")
print("  - detection_val_augmented.csv")
print("  - detection_test_augmented.csv")

print("\nSample of augmented training data:")
print(augmented_train_df.head(10))