import pandas as pd

from classifier import features, household_rule
from importers import ensure_ids

HOUSEHOLD = {"account": "Haushalt", "ibans": ["DE00 1111 2222"], "members": ["muster"],
             "contribution_keywords": ["haushaltskonto"], "contribution_category": "Haushaltskonto", "cash_category": "Bargeld"}
PREDICTION = [("Essen", 0.7), ("Lebensmittel", 0.3)]  # model predictions never contain rule categories


def rule(account, amount, partner, reference, category, iban=""):
    df = pd.DataFrame([{"account": account, "amount.value": amount, "partnerName": partner, "reference": reference,
                        "partnerAccount.iban": iban}])
    feature = next(features(df).itertuples())
    return household_rule(df.iloc[0], feature, category, PREDICTION, HOUSEHOLD)


def test_contribution_in_household_account():
    assert rule("Haushalt", 800, "Max Muster", "Haushaltskonto 01.24", "Bargeld")[0] == "Haushaltskonto"


def test_other_payments_into_household_account_may_stay_contributions():
    assert rule("Haushalt", 300, "Max Muster", "Aufstockung", "Haushaltskonto") is None


def test_reimbursement_from_household_gets_real_purpose():
    assert rule("Haushalt", -26, "Max Muster", "Heuriger", "Bargeld")[0] == "Essen"


def test_cash_withdrawal_stays_cash():
    assert rule("Haushalt", -100, "", "Auszahlung Geldautomat", "Lebensmittel")[0] == "Bargeld"
    assert rule("Giro", -50, "", "ATM 50,00 AT K1 19.05.", "Essen")[0] == "Bargeld"


def test_own_account_transfers_with_household():
    # reimbursement from the household is cash in the own account, offsetting the own cash withdrawals
    assert rule("Giro", 26, "Haushalt", "Heuriger", "Essen", iban="DE0011112222")[0] == "Bargeld"
    assert rule("Giro", -800, "Haushalt", "Haushaltskonto", "Lebensmittel", iban="DE00 1111 2222")[0] == "Haushaltskonto"
    assert rule("Giro", -20, "Shop", "Einkauf", "Lebensmittel") is None


def test_ensure_ids_keeps_existing_and_numbers_new_rows():
    df = ensure_ids(pd.DataFrame({"id": [3, None, 1, None]}))
    assert df["id"].tolist() == [3, 4, 1, 5]
    assert ensure_ids(pd.DataFrame({"a": [1, 2]}))["id"].tolist() == [1, 2]


def test_fixed_category():
    from classifier import fixed_category
    rules = [{"pattern": "amazon|amzn", "category": "Shopping"}]
    assert fixed_category("amazon payments europe 303-123 amzn mktp de", rules) == "Shopping"
    assert fixed_category("billa dankt", rules) is None


def test_member_transfers_in_household_account():
    from classifier import is_member_transfer
    df = pd.DataFrame([{"account": "Haushalt", "amount.value": -31, "partnerName": "Max Muster", "reference": "Bouldern",
                        "partnerAccount.iban": "AT00"}])
    assert is_member_transfer(df.iloc[0], next(features(df).itertuples()), HOUSEHOLD)
    df.loc[0, "account"] = "Giro"
    assert not is_member_transfer(df.iloc[0], next(features(df).itertuples()), HOUSEHOLD)


def test_member_named_only_in_reference_is_no_member_transfer():
    from classifier import is_member_transfer
    df = pd.DataFrame([{"account": "Haushalt", "amount.value": -210, "partnerName": "Tennis Academy",
                        "reference": "Tenniscamp Samuel Muster", "partnerAccount.iban": ""}])
    assert not is_member_transfer(df.iloc[0], next(features(df).itertuples()), HOUSEHOLD)
