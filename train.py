import csv
import os
import fasttext
import pandas as pd
from sharedfunctions import prep_fasttext

reldir = "model"
absdir = os.path.join(os.path.dirname(__file__), reldir)

# TODO get training file from argument instead

# history_filename = "historical_transactions_c copy.xlsx"
history_filename = "all_statements.xlsx"
with open(history_filename) as hist_file:
    hist_df = pd.read_excel(history_filename, sheet_name="Sheet1")


prep_df = hist_df[hist_df.category.notnull()]

# prune data while treating  all columns as string
# this is required for compatibility issues with old historical DB files
prep_df = prep_fasttext(prep_df)

# print(prep_df["fasttext"])
cutoff = int(round((prep_df.shape[0]) * 0.8))
prep_df = prep_df.sample(frac=1)

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
