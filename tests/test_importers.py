import json

import pandas as pd
import pytest

from config import CONFIG
from importers import combine_statements, new_transactions, read_statement

DKB_NEW = '''﻿"Girokonto";"DE00120300000000000000"
"Zeitraum:";"01.01.2025 - 31.12.2025"
"Kontostand vom 31.12.2025:";"1.000,00 €"
""
"Buchungsdatum";"Wertstellung";"Status";"Zahlungspflichtige*r";"Zahlungsempfänger*in";"Verwendungszweck";"Umsatztyp";"IBAN";"Betrag (€)";"Gläubiger-ID";"Mandatsreferenz";"Kundenreferenz"
"03.01.25";"03.01.25";"Gebucht";"ISSUER";"Bäckerei Müller";"VISA Debitkartenumsatz vom 02.01.2025";"Ausgang";"DE00000000000000000001";"-1.234,56";"";"";""
"03.01.25";"03.01.25";"Gebucht";"Erika Muster";"Max Muster";"Haushalt";"Eingang";"DE00000000000000000002";"800";"";"";""
"03.01.25";"03.01.25";"Gebucht";"Hans Muster";"Max Muster";"Haushalt";"Eingang";"DE00000000000000000003";"800";"";"";""
"04.01.25";"04.01.25";"Vorgemerkt";"ISSUER";"Shop";"VISA Debitkartenumsatz";"Ausgang";"";"-5";"";"";""
'''

DKB_OLD = '''"Kontonummer:";"DE00 1203 0000 0000 0000 00 / Girokonto";

"Von:";"01.01.2023";
"Bis:";"31.01.2023";
"Kontostand vom 31.01.2023:";"1.000,00 EUR";

"Buchungstag";"Wertstellung";"Buchungstext";"Auftraggeber / Begünstigter";"Verwendungszweck";"Kontonummer";"BLZ";"Betrag (EUR)";"Gläubiger-ID";"Mandatsreferenz";"Kundenreferenz";
"05.01.2023";"05.01.2023";"Lastschrift";"Stromfirma Süd";"Abschlag";"DE00111122223333444455";"BYLADEM1001";"-1.234,56";"";"";"";
"02.01.2023";"02.01.2023";"Tagessaldo";"";"";"";"";"0,00";"";"";"";
'''

DKB_CREDIT = '''"Kreditkarte:";"4748********1234 Kreditkarte";
"Von:";"01.01.2022";
"Bis:";"31.01.2022";
"Saldo:";"-100,00 EUR";
"Umsatz abgerechnet und nicht im Saldo enthalten";"Wertstellung";"Belegdatum";"Beschreibung";"Betrag (EUR)";"Ursprünglicher Betrag";
"Ja";"05.01.22";"04.01.22";"Beispielshop";"-1.049,90";"";
'''

CARDCOMPLETE = '''Kartenumsaetze
DATUM-DATE,HAENLDERNAME-MERCHANT_NAME,BETRAG-AMOUNT,WAEHRUNG-CURRENCY,KARTENNUMMER-CARD_NUMBER
15.02.2021,Beispiel Merchant,"12,50",EUR,4548********0000
'''

SPARKASSE = [{"booking": "2024-03-11T00:00:00.000+0100", "partnerName": "Beispiel", "reference": "Test",
              "partnerAccount": {"iban": "AT000000000000000000"}, "amount": {"value": -600000, "precision": 2, "currency": "EUR"}}]


def write(tmp_path, name, content, encoding="utf-8"):
    file = tmp_path / name
    file.write_bytes(content.encode(encoding))
    return str(file)


def test_dkb_new_format(tmp_path):
    df = read_statement(write(tmp_path, "export.csv", DKB_NEW))
    assert df.attrs["account"] == CONFIG["accounts"]["dkb_giro"]
    assert df.attrs["period"] == ("2025-01-01", "2025-12-31")
    assert len(df) == 3  # pending transaction skipped
    assert df["amount.value"].tolist() == [-1234.56, 800.0, 800.0]
    # payee for outgoing, payer for incoming transactions
    assert df["partnerName"].tolist() == ["Bäckerei Müller", "Erika Muster", "Hans Muster"]
    assert (df["booking"] == "2025-01-03").all()


def test_dkb_old_format_latin1(tmp_path):
    df = read_statement(write(tmp_path, "old.csv", DKB_OLD, "latin_1"))
    assert df.attrs["period"] == ("2023-01-01", "2023-01-31")
    assert len(df) == 1  # Tagessaldo skipped
    row = df.iloc[0]
    assert (row["booking"], row["partnerName"], row["amount.value"]) == ("2023-01-05", "Stromfirma Süd", -1234.56)
    assert row["partnerAccount.iban"] == "DE00111122223333444455"


def test_dkb_credit(tmp_path):
    df = read_statement(write(tmp_path, "credit.csv", DKB_CREDIT, "latin_1"))
    assert df.attrs["account"] == CONFIG["accounts"]["dkb_credit"]
    assert (df.iloc[0]["booking"], df.iloc[0]["amount.value"]) == ("2022-01-04", -1049.9)


def test_cardcomplete(tmp_path):
    df = read_statement(write(tmp_path, "cards.csv", CARDCOMPLETE))
    assert (df.iloc[0]["booking"], df.iloc[0]["amount.value"], df.iloc[0]["account"]) == ("2021-02-15", 12.5, "4548********0000")


def test_sparkasse_and_empty_export(tmp_path):
    df = read_statement(write(tmp_path, "AT00_2024-01-01_2024-12-31.json", json.dumps(SPARKASSE)))
    assert (df.iloc[0]["booking"], df.iloc[0]["amount.value"]) == ("2024-03-11", -6000.0)
    assert df.attrs["period"] == ("2024-01-01", "2024-12-31")
    empty = read_statement(write(tmp_path, "AT00_2019-01-01_2019-12-31.json", "[]"))
    assert empty.empty and empty.attrs["period"] == ("2019-01-01", "2019-12-31")
    assert empty.attrs["account"] == CONFIG["accounts"]["sparkasse"]


def test_unknown_files_are_ignored(tmp_path):
    assert read_statement(write(tmp_path, "notes.csv", "a;b\n1;2\n")) is None
    assert read_statement(write(tmp_path, "other.json", '{"a": 1}')) is None
    assert read_statement(write(tmp_path, "readme.txt", "text")) is None


def frame(rows):
    return pd.DataFrame(rows, columns=["account", "booking", "amount.value", "partnerName", "reference"])


def test_identical_transactions_on_same_day_are_kept():
    existing = frame([["A", "2025-01-03", 800.0, "Erika", "Haushalt"]])
    candidates = frame([["A", "2025-01-03", 800.0, "Erika", "Haushalt"], ["A", "2025-01-03", 800.0, "Hans", "Haushalt"]])
    new = new_transactions(existing, candidates)
    assert new["partnerName"].tolist() == ["Hans"]


def test_changed_texts_do_not_duplicate():
    existing = frame([["A", "2025-06-30", -120.0, "PIZZERIA.AL/CAMPO", "VISA Debitkartenumsatz"]])
    candidates = frame([["A", "2025-06-30", -120.0, "Pizzeria", "VISA Debitkartenumsatz vom 28.06.2025"]])
    assert new_transactions(existing, candidates).empty


def test_combine_overlapping_exports():
    a = frame([["A", "2025-01-02", -5.0, "x", "1"], ["A", "2025-01-03", -7.0, "y", "2"]])
    b = frame([["A", "2025-01-03", -7.0, "y", "2 new text"], ["A", "2025-01-04", -9.0, "z", "3"]])
    combined = combine_statements([a, b])
    assert sorted(combined["amount.value"]) == [-9.0, -7.0, -5.0]
    assert combined.index.is_unique
