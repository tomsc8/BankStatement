import pandas as pd

from sharedfunctions import clean_text, prep_fasttext


def test_clean_text():
    assert clean_text("BILLA DANKT 0002050/BADEN") == "billa dankt"
    assert clean_text("Wohnbauförderung Land NÖ 27.04.") == "wohnbauforderung land"  # "no" is an english stopword
    assert clean_text("Stoecklalm m14 VISA Debit") == "stoecklalm m14"
    assert clean_text("Straße") == "strasse"


def test_prep_fasttext_labels_only_categorized_rows():
    df = pd.DataFrame({"partnerName": ["Billa", None], "reference": ["Einkauf", "Strom"], "category": ["Lebensmittel", None]})
    out = prep_fasttext(df)["fasttext"].tolist()
    assert out == ["__label__Lebensmittel billa einkauf", "strom"]
