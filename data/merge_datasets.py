import pandas as pd
from sklearn.model_selection import train_test_split

def remove_duplicates(df):
    initial_size = len(df)
    df = df.drop_duplicates(subset='text', keep='first').reset_index(drop=True)
    print(f"Removed {initial_size - len(df)} duplicates")
    return df

def merge_detection_data(df1, df2, df3, df4):
    dfs = [df1, df2, df3, df4]
    
    for i, df in enumerate(dfs):
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"Dataset {i+1} missing required columns")
    
    merged = pd.concat(dfs, ignore_index=True)
    merged = remove_duplicates(merged)
    merged = merged.dropna().reset_index(drop=True)
    
    print(f"Merged dataset size: {len(merged)}")
    print(f"Class distribution:\n{merged['label'].value_counts()}")
    return merged

def balance_classes(df, method='undersample'):
    class_counts = df['label'].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    minority_size = class_counts[minority_class]
    
    if method == 'undersample':
        minority_df = df[df['label'] == minority_class]
        majority_df = df[df['label'] == majority_class].sample(n=minority_size, random_state=42)
        balanced = pd.concat([minority_df, majority_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Balanced dataset size: {len(balanced)}")
        return balanced
    
    elif method == 'weights':
        total = len(df)
        weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
        print(f"Class weights: {weights}")
        return weights
    
    elif method == 'oversample':
        minority_df = df[df['label'] == minority_class]
        majority_df = df[df['label'] == majority_class]
        minority_oversampled = minority_df.sample(n=len(majority_df), replace=True, random_state=42)
        balanced = pd.concat([minority_oversampled, majority_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Balanced dataset size: {len(balanced)}")
        return balanced
    
    raise ValueError(f"Unknown method: {method}")

def train_val_test_split(df, train_size=0.7, val_size=0.15, test_size=0.15):
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Sizes must sum to 1.0")
    
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=42
    )
    
    val_ratio = val_size / (train_size + val_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val['label'], random_state=42
    )
    
    print(f"Train size: {len(train)}")
    print(f"Val size: {len(val)}")
    print(f"Test size: {len(test)}")
    
    return train, val, test