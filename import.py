import json
import os
import re

import fasttext
import numpy as np
import pandas as pd
from sharedfunctions import prep_fasttext

# specify filename which holds complete history of transaction data
history_filename = "historical_json.xlsx"

inputdir = "input"
modeldir = "model"
absdir = os.path.join(os.path.dirname(__file__), inputdir)
inputfiles = []

# scan through folder /reldir and create a list of all identified .json files
for filename in os.listdir(absdir):
    f = os.path.join(absdir, filename)
    if os.path.isfile(f) and f.endswith('.json'):
        inputfiles.append(f)

# define with columns to use after import
FIELDS = ["booking", "partnerName", "partnerAccount.iban", "partnerAccount.bic", "partnerAccount.number",
          "partnerAccount.bankCode", "amount.value", "amount.currency", "reference"]
# define unique key to identify duplicates
KEY = ["booking", "amount.value", "reference"]

for filename in inputfiles:
    # open file and close afterwards
    with open(filename) as trx_file:
        serialdata = json.load(trx_file)

    # flatten .json structure
    file_df = pd.json_normalize(serialdata)
    try:
        input_df = pd.concat([input_df, file_df]).drop_duplicates(subset=KEY)
    except:
        input_df = file_df

input_df["amount.value"] = input_df["amount.value"].div(100).round(2)
# only keep the columns defined in FIELDS
input_df = input_df[FIELDS]
# print(input_df)
input_df = prep_fasttext(input_df)
# print(input_df["fasttext"])

# load model
model = fasttext.load_model(os.path.join(os.path.dirname(__file__), modeldir, "bs.model"))
# classify transactions
predictions = []
for line in input_df["fasttext"]:
    pred_label = model.predict(line, k=-1, threshold=0.5)
    predictions.append(pred_label)

input_df[["category", "probability"]] = predictions
input_df["category"] = input_df["category"].astype(str).str.replace("label", '').str.replace("[^a-zA-Z]", '',
                                                                                             regex=True)

print(input_df["category"])
input_df.to_excel("2022.xlsx", sheet_name="Sheet1", index=False)

# TODO
# refactor filenames into MODELPATH constants
# delete fasttext column again
# write to excel or to historical exel

# TODO add new transactions to existing historical DB
# with open(history_filename) as hist_file:
#     hist_df = pd.read_excel(history_filename, sheet_name="Sheet1")
# df = pd.concat([hist_df, input_df]).drop_duplicates(subset=KEY)
# df.to_excel(history_filename, sheet_name="Sheet1", index=False)


# bugs
# empty column reference does not work as proper unique key -> line gets duplicated. Maybe typecast problem?
# 2022-01-10T00:00:00.000+0100	Johanna Fersterer	AT902022700401024773			20227	-15	EUR
# 2022-01-10T00:00:00.000+0100	Johanna Fersterer	AT902022700401024773			20227	-15	EUR

# next steps
# cleaning function universal for all DFs historical + new
# don't write cleaning function to file
# train model
