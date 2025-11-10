"""
Dataset loading functions.

Loads datasets from HuggingFace Hub:
- ieuniversity/flirty_or_not
- aidanzhou/tinderflirting  
- ieuniversity/neutral_to_flirty
"""

from datasets import load_dataset
import pandas as pd


def load_flirty_or_not():
    """
    Load ieuniversity/flirty_or_not dataset.
    
    Returns:
        pd.DataFrame: DataFrame with 'text' and 'label' columns
    """
    pass


def load_tinderflirting():
    """
    Load aidanzhou/tinderflirting dataset.
    
    Returns:
        pd.DataFrame: DataFrame with 'text' and 'label' columns
    """
    pass


def load_neutral_to_flirty():
    """
    Load ieuniversity/neutral_to_flirty dataset.
    
    Returns:
        pd.DataFrame: DataFrame with 'non-flirty' and 'flirty' columns
    """
    pass
