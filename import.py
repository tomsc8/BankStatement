import json
import os
import re

import fasttext
import numpy as np
import pandas as pd
from sharedfunctions import prep_fasttext

# TODO import .csv from cardcomplete, DKB, DKB creditcard and name account as additional column

# specify filename which holds complete history of transaction data
history_filename = "historical_json.xlsx"

inputdir = "input"
modeldir = "model"
absdir = os.path.join(os.path.dirname(__file__), inputdir)
inputfiles = []

# scan through folder /reldir and create a list of all identified .json files
for filename in os.listdir(absdir):
    f = os.path.join(absdir, filename)
    if os.path.isfile(f) and f.endswith('.csv'):
        inputfiles.append(f)

# define with columns to use after import
FIELDS = ["booking", "partnerName", "amount.value", "amount.currency", "reference"]
# "partnerAccount.iban", "partnerAccount.bic", "partnerAccount.number", "partnerAccount.bankCode",

# define unique key to identify duplicates
KEY = ["booking", "amount.value", "reference"]

for filename in inputfiles:
    # open file and close afterwards
    with open(filename, encoding='latin_1') as trx_file:
        if filename.endswith('.json'):
            # TODO json to dataframe function
            file_df = pd.json_normalize(json.load(trx_file))
            file_df["amount.value"] = file_df["amount.value"].div(100).round(2)
        if filename.endswith('.csv') and '10527' in filename:
            file_df = pd.read_csv(trx_file, delimiter=';', header=4, quoting=1, decimal=',', thousands='.',
                                  parse_dates=["Buchungstag"])
            file_df.rename(
                columns={'Auftraggeber / Begünstigter': "partnerName", 'Betrag (EUR)': 'amount.value', "Buchungstag": "booking",
                         "Verwendungszweck": "reference"}, inplace=True)
            file_df.insert(4, "amount.currency", "EUR")

        if filename.endswith('.csv') and '4748' in filename:
            # DKB Kreditkarte Importer
            file_df = pd.read_csv(trx_file, delimiter=';', header=4, quoting=1, decimal=',', thousands='.',
                                  parse_dates=["Belegdatum"])
            file_df.rename(columns={'Betrag (EUR)': 'amount.value', "Belegdatum": "booking",
                                    "Beschreibung": "reference"}, inplace=True)
            file_df.insert(4, "amount.currency", "EUR")
            file_df.insert(5, "partnerName", "")

        if filename.endswith('.csv') and 'transactions' in filename:
            # card complete importer
            file_df = pd.read_csv(trx_file, skiprows=1, decimal=',', thousands='.', parse_dates=["DATUM-DATE"])
            file_df.rename(columns={"HAENLDERNAME-MERCHANT_NAME": "partnerName", 'BETRAG-AMOUNT': 'amount.value',
                                    "WAEHRUNG-CURRENCY": "amount.currency", "DATUM-DATE": "booking",
                                    "BRANCHE-CATEGORY": "reference"}, inplace=True)

    file_df = file_df[FIELDS]

    try:
        input_df = pd.concat([input_df, file_df]).drop_duplicates(subset=KEY)
    except:
        input_df = file_df

input_df = prep_fasttext(input_df)
print(input_df["fasttext"])

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
input_df.to_excel("2022_test.xlsx", sheet_name="Sheet1", index=False)

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
