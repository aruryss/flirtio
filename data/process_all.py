import pandas as pd
import numpy as np
from preprocessor import clean_text, tokenize_text, create_sequences, pad_sequences
from merge_datasets import merge_detection_data, balance_classes, train_val_test_split
from vocabulary import build_vocab, add_special_tokens, save_vocab

def process_detection_data(df1, df2, df3):
    print("\n=== Processing Detection Data ===")
    
    merged = merge_detection_data(df1, df2, df3)
    balanced = balance_classes(merged, method='undersample')
    train, val, test = train_val_test_split(balanced)
    
    train['text'] = train['text'].apply(clean_text)
    val['text'] = val['text'].apply(clean_text)
    test['text'] = test['text'].apply(clean_text)
    
    train.to_csv('data/processed/detection_train.csv', index=False)
    val.to_csv('data/processed/detection_val.csv', index=False)
    test.to_csv('data/processed/detection_test.csv', index=False)

def process_generation_data(gen_df):
    print("\n=== Processing Generation Data ===")
    
    gen_df = gen_df.dropna().reset_index(drop=True)
    gen_df['non-flirty'] = gen_df['non-flirty'].apply(clean_text)
    gen_df['flirty'] = gen_df['flirty'].apply(clean_text)
    
    encoder_texts = gen_df['non-flirty'].tolist()
    decoder_texts = gen_df['flirty'].tolist()
    
    encoder_vocab = build_vocab(encoder_texts, min_freq=1, max_size=5000)
    decoder_vocab = build_vocab(decoder_texts, min_freq=1, max_size=5000)
    
    encoder_vocab = add_special_tokens(encoder_vocab)
    decoder_vocab = add_special_tokens(decoder_vocab)
    
    save_vocab(encoder_vocab, 'data/processed/encoder_vocab.pkl')
    save_vocab(decoder_vocab, 'data/processed/decoder_vocab.pkl')
    
    encoder_sequences = create_sequences(encoder_texts, encoder_vocab)
    decoder_sequences = create_sequences(decoder_texts, decoder_vocab)
    
    max_encoder_len = max(len(seq) for seq in encoder_sequences)
    max_decoder_len = max(len(seq) for seq in decoder_sequences)
    
    print(f"Max encoder length: {max_encoder_len}")
    print(f"Max decoder length: {max_decoder_len}")
    
    encoder_padded = pad_sequences(encoder_sequences, maxlen=max_encoder_len, padding='post')
    decoder_padded = pad_sequences(decoder_sequences, maxlen=max_decoder_len, padding='post')
    
    np.save('data/processed/encoder_sequences.npy', encoder_padded)
    np.save('data/processed/decoder_sequences.npy', decoder_padded)
    
    with open('data/processed/max_lengths.txt', 'w') as f:
        f.write(f"max_encoder_len={max_encoder_len}\n")
        f.write(f"max_decoder_len={max_decoder_len}\n")
    
    print("✓ Generation data saved")
    
if __name__ == "__main__":
    print("Loading datasets...")
    
    df1 = pd.read_csv('data\\raw\\emojiflirting.csv')
    df2 = pd.read_csv('data\\raw\\swipestatsflirting.csv')
    df3 = pd.read_csv('data\\raw\\tinderflirting.csv')
    
    process_detection_data(df1, df2, df3)
    
    gen_df = pd.read_csv('data\\raw\\generation.csv')
    process_generation_data(gen_df)
    
    print("\n✓✓✓ All data processing complete! ✓✓✓")
