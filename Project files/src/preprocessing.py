"""
Preprocessing and Cleaning Module for Context-Aware Bangla Text Analyzer.
Implements Lab 1 (Regular Expressions & Text Preprocessing).
Lead: Md. Tariful Islam Jony (ID: 2107119)
"""

import re
import unicodedata
from typing import Optional


def clean_bangla_text(text: Optional[str]) -> str:
    """
    Cleans raw Bangla / Banglish social media text using regex and normalization:
    - Handles None/NaN values
    - Strips HTML tags, URLs, and social mentions
    - Removes Zero-Width Non-Joiners (ZWNJ) and invisible control characters
    - Normalizes Unicode characters to NFC
    - Preserves Bengali script, English words, numbers, and expressive punctuation (! ? । , .)
    - Collapses repeated whitespaces
    """
    if not isinstance(text, str):
        return ""

    # Normalize Unicode form
    text = unicodedata.normalize("NFC", text)

    # Remove BOM / Zero-width characters
    text = re.sub(r'[\ufeff\u200b\u200c\u200d\u200e\u200f\u00ad]', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove social mentions (@user) and hashtags (#hashtag -> keep word if needed, or remove #)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Retain Bengali (\u0980-\u09FF), English (a-zA-Z), Digits (0-9, \u09E6-\u09EF),
    # and emotionally salient punctuation (! ? . , । -)
    allowed_pattern = r'[^\u0980-\u09FFa-zA-Z0-9\u09E6-\u09EF\s!?.,।\-"\']'
    text = re.sub(allowed_pattern, ' ', text)

    # Normalize multiple punctuation (e.g., '!!!!' -> '!!', '????' -> '??', '.....' -> '...')
    text = re.sub(r'!{2,}', '!!', text)
    text = re.sub(r'\?{2,}', '??', text)
    text = re.sub(r'\.{2,}', '...', text)
    text = re.sub(r'।{2,}', '।', text)

    # Collapse repeated whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


if __name__ == "__main__":
    sample = "<html>বাহ! কী অসাধারণ service... তিন ঘণ্টা অপেক্ষাও করেও কাজ হলো না! http://test.com #bad_service"
    cleaned = clean_bangla_text(sample)
    print("Original:", sample)
    print("Cleaned: ", cleaned)
