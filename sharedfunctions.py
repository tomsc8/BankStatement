import texthero as hero
from texthero import preprocessing, stopwords

def prep_fasttext(dfp):
    # custom stopwords
    default_stopwords = stopwords.DEFAULT
    custom_stopwords = default_stopwords.union({"k1", "e", "comm", "paypal", "nan", "k2", "karte2", "um", "none", "eu", "wien", "baden", "at", "ag", "de", "pos"})
    # pre-processing
    custom_pipeline = [preprocessing.fillna,
                       preprocessing.lowercase,
                       preprocessing.remove_digits,
                       preprocessing.remove_diacritics,
                       # preprocessing.remove_brackets
                       preprocessing.remove_punctuation,
                       preprocessing.remove_whitespace,
                       ]

    dfp['fasttext'] = hero.clean(dfp["partnerName"].astype('str') + " " + dfp["reference"].astype('str'), custom_pipeline)
    dfp['fasttext'] = hero.remove_stopwords(dfp['fasttext'], custom_stopwords)

    # TODO
    #  if label then with __label__ if no label then empty filter out all rows that don't have a category assigned.
    #  atm is only checked for full column availability but it should be on cell level.
    if 'category' in dfp.columns:
        dfp['fasttext'] = "__label__" + dfp['category'] + " " + dfp['fasttext']
    return dfp