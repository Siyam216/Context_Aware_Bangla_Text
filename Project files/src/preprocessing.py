"""
Preprocessing, Tokenization and Lexical Module for Context Aware Bangla Text Analyzer.
Implements:
- Lab 1: Regular Expressions, Text Cleaning & Bengali Word Tokenization.
- Lab 2: N-gram generation and Stop-words filtering with negation preservation.
"""

import re
import unicodedata
from typing import Optional, List, Tuple, Set


# Standard Bengali Stop Words
DEFAULT_BANGLA_STOPWORDS: Set[str] = {
    'অতএব', 'অথচ', 'অথবা', 'অনুযায়ী', 'অনেক', 'অনেকে', 'অনেকেই', 'অবশ্য', 'অবশেষে',
    'অর্থাত', 'আইন', 'আগে', 'আগেই', 'আছে', 'আজ', 'আদ্যোপান্ত', 'আপনার', 'আপনি',
    'আবার', 'আমরা', 'আমাকে', 'আমাদের', 'আমার', 'আমি', 'আর', 'আরও', 'ইত্যাদি',
    'ইহা', 'উচিত', 'উত্তর', 'উপাংশ', 'উপরে', 'উপরেও', 'উভয়', 'উভয়েই', 'উল্টা',
    'একটি', 'একবার', 'একে', 'একই', 'একজন', 'এত', 'এতে', 'এদের', 'এমন', 'এমনকি',
    'এমনি', 'এর', 'এরা', 'এল', 'এস', 'এসে', 'ঐ', 'ও', 'ওই', 'ওকে', 'ওখানে',
    'ওদের', 'ওর', 'ওরা', 'কখন', 'কত', 'কবে', 'কম', 'কয়েক', 'কয়েকটি', 'করা',
    'করবে', 'করলেন', 'করছেন', 'করে', 'করেই', 'করেছেন', 'করছে', 'করবেন', 'করলে',
    'করার', 'করেছিলেন', 'করলে', 'করাই', 'কাউকে', 'কাছে', 'কারও', 'কারণ', 'কাজে',
    'কী', 'কে', 'কেউ', 'কেউই', 'কেন', 'কোটি', 'কোথায়', 'কোন', 'কোনও', 'কোনো',
    'গেল', 'গেলেন', 'গেছে', 'গেছেন', 'যখন', 'যদি', 'যদিও', 'যাকে', 'যা', 'যায়',
    'যাদের', 'যার', 'যারা', 'যে', 'যেখানে', 'যেতে', 'যেন', 'যেমন', 'থাকবে',
    'থাকবেন', 'থাকলে', 'থাকে', 'থাকেন', 'থেকে', 'থেকেই', 'তখন', 'তত', 'তবে',
    'তা', 'তাঁকে', 'তাঁদের', 'তাঁর', 'তাঁরা', 'তাই', 'তাও', 'তাকে', 'তাদের',
    'তার', 'তারই', 'তারা', 'তালিকা', 'তিমির', 'তিনি', 'তুলে', 'তুমি', 'তোমার',
    'তোমাদের', 'তোমাকে', 'তো', 'দিয়ে', 'দিয়েছেন', 'দিলেন', 'দুটি', 'দুটো', 'দেওয়া',
    'দেওয়ার', 'দেখা', 'দেখে', 'দেখেছেন', 'দেন', 'দেয়', 'দ্বারা', 'ধরে', 'ধরা',
    'নতুন', 'নানা', 'নিজে', 'নিজেই', 'নিজেদের', 'নিজের', 'নিয়ে', 'পর', 'পরে',
    'পরেই', 'পারি', 'পারেন', 'প্রায়', 'প্রতি', 'প্রথম', 'প্রভৃতি', 'ফলে', 'ফের',
    'বলা', 'বললেন', 'বলেন', 'বলছেন', 'বলে', 'বলেছেন', 'বলল', 'বহু', 'বা', 'বাদে',
    'বার', 'বেশ', 'বোধহয়', 'মতো', 'মধ্য', 'মধ্যে', 'মনে', 'মাত্র', 'মোটেই',
    'যথেষ্ট', 'সকল', 'সব', 'সবাই', 'সবাইকে', 'সমস্ত', 'সহ', 'সবার', 'সিট', 'সুতরাং',
    'সে', 'সেই', 'সেখান', 'সেখানে', 'সেটা', 'সেটাই', 'সেটাও', 'সেটি', 'স্পষ্ট',
    'স্বয়ং', 'হওয়া', 'হওয়ার', 'হচ্ছে', 'হত', 'হতে', 'হতেই', 'হবে', 'হবেন',
    'হয়', 'হয়ে', 'হয়েছিল', 'হয়েছেন', 'হলেন', 'হলে', 'হলেই', 'হলো'
}

# CRITICAL NEGATION WORDS - NEVER TO BE DROPPED
NEGATION_WORDS: Set[str] = {
    'না', 'নয়', 'নেই', 'নাহ', 'নাই', 'নহে', 'নহেক', 'না-না', 'নয়তো',
    'কখনো', 'কখনোই', 'বিনা', 'ছাড়া', 'ব্যতীত', 'বাধা', 'বঞ্চিত'
}

# Safe stop words set (ensuring negations are never included)
SAFE_STOPWORDS: Set[str] = DEFAULT_BANGLA_STOPWORDS - NEGATION_WORDS


def clean_bangla_text(text: Optional[str]) -> str:
    """
    Cleans raw Bangla / Banglish social media text using regex and normalization:
    - Strips HTML tags, URLs, and social mentions
    - Removes Zero-Width Non-Joiners (ZWNJ) and invisible control characters
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

    # Remove social mentions (@user) and hashtags (#hashtag -> keep word)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Retain Bengali, English, Digits, and expressive punctuation
    allowed_pattern = r'[^\u0980-\u09FFa-zA-Z0-9\u09E6-\u09EF\s!?.,।\-"\']'
    text = re.sub(allowed_pattern, ' ', text)

    # Normalize multiple punctuation
    text = re.sub(r'!{2,}', '!!', text)
    text = re.sub(r'\?{2,}', '??', text)
    text = re.sub(r'\.{2,}', '...', text)
    text = re.sub(r'।{2,}', '।', text)

    # Collapse repeated whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_bangla(text: str, keep_punct: bool = True) -> List[str]:
    """
    Tokenizes clean Bengali text into words and punctuation tokens.
    Handles Bengali sentence terminators ('।', '?', '!') and commas.
    Example: 'বাহ! কাজ হলো না!' -> ['বাহ', '!', 'কাজ', 'হলো', 'না', '!']
    """
    if not isinstance(text, str) or not text.strip():
        return []

    if keep_punct:
        # Separate punctuation into individual tokens
        tokens = re.findall(r'[\u0980-\u09FFa-zA-Z0-9\u09E6-\u09EF]+|[!?.,।"\'-]', text)
    else:
        tokens = re.findall(r'[\u0980-\u09FFa-zA-Z0-9\u09E6-\u09EF]+', text)

    return tokens


def remove_stopwords(tokens: List[str], preserve_negation: bool = True) -> List[str]:
    """
    Removes standard Bengali stop-words while strictly preserving negation markers.
    Preserving negation words ('না', 'নয়', 'নেই', etc.) is vital for Sentiment & Sarcasm.
    """
    active_stopwords = SAFE_STOPWORDS if preserve_negation else DEFAULT_BANGLA_STOPWORDS
    return [t for t in tokens if t not in active_stopwords]


def get_ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
    """
    Generates n-grams from a list of tokens (Lab 2 Language Modeling).
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


if __name__ == "__main__":
    sample = "বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!"
    cleaned = clean_bangla_text(sample)
    tokens = tokenize_bangla(cleaned)
    filtered = remove_stopwords(tokens, preserve_negation=True)
    bigrams = get_ngrams(filtered, n=2)

    print("Tokens:        ", tokens)
    print("Filtered:      ", filtered)
    print("Bigrams:       ", bigrams)
    print("Negation 'না' kept in filtered?", 'না' in filtered)
