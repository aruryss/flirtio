import pandas as pd
import numpy as np
from preprocessor import clean_text, create_sequences, pad_sequences
from merge_datasets import merge_detection_data, balance_classes, train_val_test_split
from vocabulary import build_vocab, add_special_tokens, save_vocab

def process_detection_data(df1, df2, df3, df4):
    print("\n=== Processing Detection Data ===")
    
    merged = merge_detection_data(df1, df2, df3, df4)
    balanced = balance_classes(merged, method='undersample')
    train, val, test = train_val_test_split(balanced)
    
    train['text'] = train['text'].apply(clean_text)
    val['text'] = val['text'].apply(clean_text)
    test['text'] = test['text'].apply(clean_text)
    
    train.to_csv('data/processed/detection_train.csv', index=False)
    val.to_csv('data/processed/detection_val.csv', index=False)
    test.to_csv('data/processed/detection_test.csv', index=False)


if __name__ == "__main__":
    print("Loading datasets...")
    
    df1 = pd.read_csv('data\\raw\\emojiflirting.csv')
    df2 = pd.read_csv('data\\raw\\neutralandflirting.csv')
    df3 = pd.read_csv('data\\raw\\swipestatsflirting.csv')
    df4 = pd.read_csv('data\\raw\\tinderflirting.csv')
    
    process_detection_data(df1, df2, df3, df4)
    
    print("\n✓✓✓ All data processing complete! ✓✓✓")
