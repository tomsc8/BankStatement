import re
import string

import pandas as pd
from unidecode import unidecode

# english stopword list of nltk, which texthero used as its default
NLTK_EN_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it",
    "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y",
    "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn",
    "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't",
    "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn",
    "wouldn't",
}

# custom stopwords
STOPWORDS = NLTK_EN_STOPWORDS | {
    "k1", "e", "comm", "paypal", "nan", "k2", "karte2", "um", "none", "eu", "wien", "baden", "at", "ag", "de", "pos",
    "debit", "visa", "debitk", "awv", "meldepflicht", "beachten", "hotline", "bundesbank", "datum", "uhr",
    "girozentrale", "tan", "uhr1", "versicherungs", "aktiengesellschaft", "folgepraemie", "folgepramie", "gmbh",
    "stripe", "via", "ppro",
}

PUNCTUATION = re.compile(rf"([{re.escape(string.punctuation)}])+")
DIGIT_BLOCKS = re.compile(r"\b\d+\b")


def clean_text(text):
    # same steps as the former texthero pipeline: lowercase, remove digit blocks, transliterate diacritics,
    # remove punctuation and stopwords, normalize whitespace
    text = DIGIT_BLOCKS.sub(" ", str(text).lower())
    text = PUNCTUATION.sub(" ", unidecode(text))
    return " ".join(word for word in text.split() if word not in STOPWORDS)


def prep_fasttext(dfp):
    text = dfp["partnerName"].fillna("").astype(str) + " " + dfp["reference"].fillna("").astype(str)
    dfp['fasttext'] = text.map(clean_text)
    # prefix rows that have a category with their label for training
    if 'category' in dfp.columns:
        has_category = dfp['category'].notna() & (dfp['category'].astype(str).str.strip() != "")
        dfp.loc[has_category, 'fasttext'] = "__label__" + dfp.loc[has_category, 'category'].astype(str) + " " + dfp.loc[has_category, 'fasttext']
    return dfp
