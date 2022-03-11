import csv
import os
import texthero as hero
import fasttext
import pandas as pd
from texthero import preprocessing, stopwords

reldir = "model"
absdir = os.path.join(os.path.dirname(__file__), reldir)

# TODO get training file from argument instead

history_filename = "historical_transactions_c copy.xlsx"
with open(history_filename) as hist_file:
    hist_df = pd.read_excel(history_filename, sheet_name="Sheet1")


def prep_fasttext(dfp):
    # custom stopwords
    default_stopwords = stopwords.DEFAULT
    custom_stopwords = default_stopwords.union({"k1", "e-comm", "paypal", "nan", "k2", "karte2", "um"})
    # pre-processing
    custom_pipeline = [preprocessing.fillna,
                       preprocessing.lowercase,
                       preprocessing.remove_digits,
                       preprocessing.remove_whitespace,
                       preprocessing.remove_diacritics,
                       # preprocessing.remove_brackets
                       ]

    # dfp['clean_text'] = dfp.astype('str')
    dfp['fasttext'] = hero.clean(dfp["partnerName"].astype('str') + " " + dfp["reference"].astype('str'), custom_pipeline)
    dfp['fasttext'] = hero.remove_stopwords(dfp['fasttext'], custom_stopwords)
    dfp['fasttext'] = hero.remove_punctuation(dfp['fasttext'])
    # TODO
    #  if label then with __label__ if no label then empty filter out all rows that don't have a category assigned.
    #  atm is only checked for full column availability but it should be on cell level.
    if 'category' in dfp.columns:
        dfp['fasttext'] = "__label__" + dfp['category'] + " " + dfp['fasttext']
    return dfp


prep_df = hist_df[hist_df.category.notnull()]

# prune data while treating  all columns as string
# this is required for compatibility issues with old historical DB files
prep_df = prep_fasttext(prep_df)

# print(prep_df["fasttext"])
cutoff = int(round((prep_df.shape[0]) * 0.8))
prep_df.sample(frac=1)

train_df = prep_df.iloc[:cutoff, :]
train_df['fasttext'].to_csv('train.txt', index=False, header=None, quoting=csv.QUOTE_NONE)

validation_df = prep_df.iloc[cutoff:, :]
validation_df['fasttext'].to_csv('validation.txt', index=False, header=None, quoting=csv.QUOTE_NONE)

# train and validate model
model = fasttext.train_supervised('train.txt', epoch=300, lr=0.8)


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


print(model.words)
print(model.labels)
print_results(*model.test('validation.txt'))

# TODO load evaluation results from existing model, compare and write model only if it has better results.

model.save_model(os.path.join(absdir, "bs.model"))
with open(os.path.join(absdir, "bs.words"), 'w') as f:
    print(model.labels, file=f)
    print(model.words, file=f)
# TODO write evaluation results into file next to model.
