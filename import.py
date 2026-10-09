import io
import json
import os
import fasttext
import pandas as pd
import numpy as np
from datetime import datetime
from sharedfunctions import prep_fasttext

# specify filename which holds complete history of transaction data
history_filename = "all_statements.xlsx"

inputdir = "input"
modeldir = "model"
absdir = os.path.join(os.path.dirname(__file__), inputdir)
inputfiles = []

# scan through folder /reldir and create a list of all identified .json files
for filename in os.listdir(absdir):
    f = os.path.join(absdir, filename)
    if os.path.isfile(f): #and f.endswith('.csv'):
        inputfiles.append(f)

# define with columns to use after import
FIELDS = ["booking", "partnerName", "partnerAccount.iban", "amount.value", "amount.currency", "reference", "account", ]
# , "partnerAccount.bic", "partnerAccount.number", "partnerAccount.bankCode",

# define unique key to identify duplicates
KEY = ["booking", "amount.value", "reference"]
german_date = lambda x: datetime.strptime(x, '%d.%m.%y')


def read_dkb_giro(filename):
    # DKB Girokonto importer for the old (Buchungstag, latin-1) and new (Buchungsdatum, utf-8) export format.
    # header position depends on the export mask, so search for the header line instead of a fixed row.
    with open(filename, 'rb') as f:
        raw = f.read()
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('latin_1')
    lines = text.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith(('"Buchungsdatum"', '"Buchungstag"')))
    df = pd.read_csv(io.StringIO("\n".join(lines[header:])), delimiter=';', quoting=1, dtype=str,
                     keep_default_na=False)

    if "Buchungsdatum" in df.columns:
        # new format: separate payer and payee columns, pending transactions marked in Status
        df = df[df["Status"] == "Gebucht"].copy()
        df.rename(columns={"Buchungsdatum": "booking", "Betrag (€)": "amount.value",
                           "IBAN": "partnerAccount.iban", "Verwendungszweck": "reference"}, inplace=True)
    else:
        # old format: single counterparty column
        df = df[~df["Buchungstext"].str.contains("Tagessaldo") & ~df["Verwendungszweck"].str.contains("Tagessaldo")].copy()
        df.rename(columns={"Buchungstag": "booking", "Betrag (EUR)": "amount.value",
                           "Auftraggeber / Begünstigter": "partnerName", "Kontonummer": "partnerAccount.iban",
                           "Verwendungszweck": "reference"}, inplace=True)

    df = df[df["booking"] != ""]
    df["amount.value"] = df["amount.value"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float)
    if "partnerName" not in df.columns:
        # counterparty is the payee for outgoing and the payer for incoming transactions
        df["partnerName"] = np.where(df["amount.value"] < 0, df["Zahlungsempfänger*in"], df["Zahlungspflichtige*r"])
    df["booking"] = pd.to_datetime(df["booking"], format="%d.%m.%y", errors="coerce").fillna(
        pd.to_datetime(df["booking"], format="%d.%m.%Y", errors="coerce"))
    df["amount.currency"] = "EUR"
    df["account"] = "DKB Konto"
    return df

for filename in inputfiles:
    # open file and close afterwards
    with open(filename, encoding='latin_1') as trx_file:
        if filename.endswith('.json'):
            # Sparkasse Importer
            file_df = pd.json_normalize(json.load(trx_file))
            file_df["amount.value"] = file_df["amount.value"].div(100).round(2)
            file_df.insert(1, "account", "Sparkasse")
            file_df["booking"] = pd.to_datetime(file_df["booking"].str.split('T').str[0], format="%Y-%m-%d")

        if filename.endswith('.csv') and '10527' in filename:
            # DKB Debitkonto Importer
            file_df = read_dkb_giro(filename)

        if filename.endswith('.csv') and '4748' in filename:
            # DKB Kreditkarte Importer
            file_df = pd.read_csv(trx_file, delimiter=';', header=4, quoting=1, decimal=',', thousands='.',
                                  parse_dates=["Belegdatum"], date_parser=german_date)
            file_df.rename(columns={'Betrag (EUR)': 'amount.value', "Belegdatum": "booking",
                                    "Beschreibung": "reference"}, inplace=True)
            # file_df["booking"] = pd.to_datetime(file_df["booking"], format="%d.%m.%Y")
            file_df.insert(4, "amount.currency", "EUR")
            file_df.insert(5, "partnerName", "")
            file_df.insert(1, "account", "DKB Kreditkarte Thomas")
            file_df.insert(6, "partnerAccount.iban", "")

        if filename.endswith('.csv') and 'transactions' in filename:
            # card complete importer
            file_df = pd.read_csv(trx_file, skiprows=1, decimal=',', thousands='.',
                                  parse_dates=["DATUM-DATE"], date_parser=german_date)
            file_df.rename(columns={"HAENLDERNAME-MERCHANT_NAME": "partnerName", 'BETRAG-AMOUNT': 'amount.value',
                                    "WAEHRUNG-CURRENCY": "amount.currency", "DATUM-DATE": "booking",
                                    "KARTENNUMMER-CARD_NUMBER": "account"}, inplace=True)
            file_df["booking"] = pd.to_datetime(file_df["booking"], format="%d.%m.%Y")
            file_df.insert(5, "reference", "")
            file_df.insert(6, "partnerAccount.iban", "")

        if filename.endswith('.xlsx'):
            # historical file re-importer
            file_df = pd.read_excel(filename, sheet_name="Sheet1")
            bool_series = pd.isnull(file_df["category"])
            file_df = file_df[bool_series]


    file_df = file_df[FIELDS]
    # convert datetime to str for writing to csv and to identify duplicates later on
    file_df["booking"] = file_df["booking"].astype('str')

    try:
        input_df = pd.concat([input_df, file_df]).drop_duplicates(subset=KEY, keep='first')
        # TODO: if one element of KEY is empty it leaves a duplicate transaction
    except:
        input_df = file_df

try:
    input_df.sort_values(["booking"], inplace=True)
except NameError:
    print("no suitable files in /input")

input_df = prep_fasttext(input_df)
print(input_df["fasttext"])

# load model
model = fasttext.load_model(os.path.join(os.path.dirname(__file__), modeldir, "bs.model"))
# classify transactions

def predict(row):
    return model.predict(row["fasttext"], k=-1, threshold=0.5)[0]

input_df["category"] = input_df.apply(predict, axis=1)

input_df["category"] = input_df["category"].astype(str).str.replace("label", '').str.replace("[^a-zA-Z]", '',
                                                                                             regex=True)

print(input_df["category"])
input_df.to_excel("new_import.xlsx", sheet_name="Sheet1", index=False)

# TODO schauen ob andere sheets existieren und diese auch speichern und schreiben.
# TODO warnings bereinigen
# TODO test ob für andere Files unnötig durchlaufen und ob Duplikate kommen
# TODO refactor filenames into MODELPATH constants

# add new transactions to existing historical DB
with open(history_filename) as hist_file:
    hist_df = pd.read_excel(history_filename, sheet_name="Sheet1")
df = pd.concat([hist_df, input_df]).drop_duplicates(subset=KEY)
df.sort_values(["booking", "account", "amount.value", "reference"], inplace=True)
df.to_excel(history_filename, sheet_name="Sheet1", index=False)

# bugs
# empty column reference does not work as proper unique key -> line gets duplicated. Maybe typecast problem?
# 2022-01-10T00:00:00.000+0100	REDACTED	XX00000000000000000000			20227	-15	EUR
# 2022-01-10T00:00:00.000+0100	REDACTED	XX00000000000000000000			20227	-15	EUR
