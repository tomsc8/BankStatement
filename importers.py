import difflib
import io
import json
import os
import re

import numpy as np
import pandas as pd

from config import CONFIG

# define with columns to use after import
FIELDS = ["booking", "partnerName", "partnerAccount.iban", "amount.value", "amount.currency", "reference", "account", ]

# transactions are matched on account, booking date and amount only. reference and partner texts change whenever
# a bank changes its export format, so they are only used to pick the best candidate within a match group.
KEY = ["account", "booking", "amount.value"]


def parse_german_date(series):
    # exports use both dd.mm.yy and dd.mm.yyyy
    return pd.to_datetime(series, format="%d.%m.%y", errors="coerce").fillna(
        pd.to_datetime(series, format="%d.%m.%Y", errors="coerce"))


def parse_german_amount(series):
    return series.str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float)


def read_text(filename):
    with open(filename, 'rb') as f:
        raw = f.read()
    try:
        return raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        return raw.decode('latin_1')


def header_index(lines, *columns):
    # index of the first line that contains all given column names, None if there is none
    return next((i for i, line in enumerate(lines) if all(c in line for c in columns)), None)


def csv_period(lines):
    # statement period from the csv preamble: "Zeitraum:";"01.01.2025 - 31.12.2025" or "Von:";".." / "Bis:";".."
    head = "\n".join(lines[:10])
    found = re.search(r'Zeitraum:";"(\d\d\.\d\d\.\d{4}) - (\d\d\.\d\d\.\d{4})', head)
    if not found:
        found = re.search(r'Von:";"(\d\d\.\d\d\.\d{4})".*\n"Bis:";"(\d\d\.\d\d\.\d{4})', head)
    if not found:
        return None
    return tuple(pd.to_datetime(d, format="%d.%m.%Y").strftime("%Y-%m-%d") for d in found.groups())


def read_sparkasse(filename, data):
    # Sparkasse (George) json export
    found = re.search(r"_(\d{4}-\d\d-\d\d)_(\d{4}-\d\d-\d\d)", os.path.basename(filename))
    period = found.groups() if found else None
    if not data:
        return pd.DataFrame(columns=FIELDS), period
    df = pd.json_normalize(data)
    df["amount.value"] = df["amount.value"].div(100).round(2)
    df["account"] = CONFIG["accounts"]["sparkasse"]
    df["booking"] = pd.to_datetime(df["booking"].str.split('T').str[0], format="%Y-%m-%d")
    return df, period


def read_dkb_giro(lines, header):
    # DKB Girokonto importer for the old (Buchungstag, latin-1) and new (Buchungsdatum, utf-8) export format
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
    df["amount.value"] = parse_german_amount(df["amount.value"])
    if "partnerName" not in df.columns:
        # counterparty is the payee for outgoing and the payer for incoming transactions
        df["partnerName"] = np.where(df["amount.value"] < 0, df["Zahlungsempfänger*in"], df["Zahlungspflichtige*r"])
    df["booking"] = parse_german_date(df["booking"])
    df["amount.currency"] = "EUR"
    df["account"] = CONFIG["accounts"]["dkb_giro"]
    return df


def read_dkb_credit(lines, header):
    # DKB Kreditkarte importer
    df = pd.read_csv(io.StringIO("\n".join(lines[header:])), delimiter=';', quoting=1, dtype=str,
                     keep_default_na=False)
    df.rename(columns={'Betrag (EUR)': 'amount.value', "Belegdatum": "booking", "Beschreibung": "reference"}, inplace=True)
    df = df[df["booking"] != ""]
    df["booking"] = parse_german_date(df["booking"])
    df["amount.value"] = parse_german_amount(df["amount.value"])
    df["amount.currency"] = "EUR"
    df["partnerName"] = ""
    df["account"] = CONFIG["accounts"]["dkb_credit"]
    df["partnerAccount.iban"] = ""
    return df


def read_cardcomplete(lines, header):
    # card complete importer, the card number is used as account
    df = pd.read_csv(io.StringIO("\n".join(lines[header:])), decimal=',', thousands='.', dtype={"DATUM-DATE": str})
    df.rename(columns={"HAENLDERNAME-MERCHANT_NAME": "partnerName", 'BETRAG-AMOUNT': 'amount.value',
                       "WAEHRUNG-CURRENCY": "amount.currency", "DATUM-DATE": "booking",
                       "KARTENNUMMER-CARD_NUMBER": "account"}, inplace=True)
    df["booking"] = parse_german_date(df["booking"])
    df["reference"] = ""
    df["partnerAccount.iban"] = ""
    return df


def read_statement(filename):
    """Read a bank export, the format is recognized by its content.

    Returns a DataFrame with FIELDS (empty for exports without transactions) and the covered period
    (first and last day, or None if unknown) and the account in df.attrs, or None for unsupported files.
    """
    period = None
    if filename.endswith('.json'):
        with open(filename, encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return None
        if not isinstance(data, list) or (data and "booking" not in data[0]):
            return None
        df, period = read_sparkasse(filename, data)
        account = CONFIG["accounts"]["sparkasse"]
    elif filename.endswith('.csv'):
        lines = read_text(filename).splitlines()
        period = csv_period(lines)
        if (i := header_index(lines, '"Buchungsdatum"', 'Zahlungsempf')) is not None or \
                (i := header_index(lines, '"Buchungstag"', 'Auftraggeber / Beg')) is not None:
            df, account = read_dkb_giro(lines, i), CONFIG["accounts"]["dkb_giro"]
        elif (i := header_index(lines, '"Belegdatum"', '"Beschreibung"')) is not None:
            df, account = read_dkb_credit(lines, i), CONFIG["accounts"]["dkb_credit"]
        elif (i := header_index(lines, 'DATUM-DATE', 'HAENLDERNAME-MERCHANT_NAME')) is not None:
            df = read_cardcomplete(lines, i)
            account = df["account"].iloc[0] if not df.empty else None
        else:
            return None
    else:
        return None

    df = normalize(df[FIELDS].copy())
    if period is None and not df.empty:
        period = (df["booking"].min(), df["booking"].max())
    df.attrs["period"] = period
    df.attrs["account"] = account
    return df


def normalize(df):
    # common representation so that transactions from different sources can be compared
    df["booking"] = pd.to_datetime(df["booking"]).dt.strftime("%Y-%m-%d")
    df["amount.value"] = df["amount.value"].astype(float).round(2)
    text_columns = ["partnerName", "partnerAccount.iban", "reference"]
    df[text_columns] = df[text_columns].fillna("").astype(str)
    return df


def text_of(row):
    text = f"{row.get('partnerName', '')} {row.get('reference', '')}".lower()
    return re.sub(r"[^a-z0-9]", "", text.replace("nan", ""))


def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def new_transactions(existing, candidates):
    """Return the rows of candidates that are not yet contained in existing.

    Rows are compared per (account, booking, amount): if existing holds n and candidates m rows for such a key,
    the m - n candidate rows that look least like the existing ones are new. Identical looking transactions on the
    same day (e.g. two transfers of the same amount) are kept, and changed export texts do not create duplicates.
    """
    if existing is None or existing.empty:
        return candidates
    existing_groups = {k: g for k, g in existing.groupby(KEY)}
    new_rows = []
    for k, group in candidates.groupby(KEY, sort=False):
        have = existing_groups.get(k)
        excess = len(group) - (0 if have is None else len(have))
        if excess <= 0:
            continue
        if have is None:
            new_rows.append(group)
            continue
        have_texts = [text_of(r) for _, r in have.iterrows()]
        best_match = group.apply(lambda r: max(similarity(text_of(r), t) for t in have_texts), axis=1)
        new_rows.append(group.loc[best_match.sort_values(kind="stable").index[:excess]])
    if not new_rows:
        return candidates.iloc[0:0]
    return pd.concat(new_rows)


def combine_statements(frames):
    # combine several exports that may overlap (e.g. two exports of the same year)
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame(columns=FIELDS)
    combined = None
    for df in frames:
        combined = df if combined is None else pd.concat([combined, new_transactions(combined, df)])
    return combined.reset_index(drop=True)
