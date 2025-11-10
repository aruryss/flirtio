"""
Dataset loading functions.
"""
import pandas as pd

def load_flirty_or_not():
    df = pd.read_csv("data\\raw\\emojiflirting.csv")
    return df

def load_tinderflirting():
    df = pd.read_csv("data\\raw\\tinderflirting.csv")
    return df

def load_swipestatsflirting():
    df = pd.read_csv("data\\raw\\swipestatsflirting.csv")
    return df

def load_neutral_to_flirty():
    df = pd.read_csv("data\\raw\\neutralandflirting.csv")
    return df

tinder_df= load_tinderflirting()
swipestats_df = load_swipestatsflirting()
withneutral_df = load_neutral_to_flirty()
withemoji_df = load_flirty_or_not()