"""Classification of transactions: history lookup, fastText model and household rules.

0. fixed: transactions matching a pattern of config "fixed_categories" get that category (e.g. marketplaces)
1. lookup: a transaction whose text, counterparty IBAN or counterparty name always had the same category in the
   history gets that category (recurring recipients do not depend on the model)
2. model: transactions of known recipients without a unique category are classified by the fastText model when it
   is sure enough. Recipients that were never categorized before are left open and grouped for review.
3. household rules (config "household"): in the household account the real purpose is booked, incoming
   contributions of the members are the contribution category and cash is only used for cash withdrawals.
   In the members' own accounts transfers to the household account are contributions and reimbursements
   from it are cash, so that they cancel out the cash payments made for the household.
"""
import re

import fasttext
import pandas as pd

from config import CONFIG
from sharedfunctions import clean_text

# below this probability the model is right only about a third of the time for recipients it has not seen,
# such transactions are left uncategorized for review
MIN_PROBABILITY = 0.9
ATM_KEYWORDS = ("auszahlung", "bankomat", "geldautomat", "bargeldbehebung", "atm ")


def features(df):
    out = pd.DataFrame(index=df.index)
    out["text"] = (df["partnerName"].fillna("").astype(str) + " " + df["reference"].fillna("").astype(str)).map(clean_text)
    out["iban"] = df["partnerAccount.iban"].fillna("").astype(str).str.replace(" ", "").str.upper()
    out["merchant"] = df["partnerName"].fillna("").astype(str).map(clean_text)
    out["raw"] = (df["partnerName"].fillna("").astype(str) + " " + df["reference"].fillna("").astype(str)).str.lower()
    out["ref"] = df["reference"].fillna("").astype(str).str.lower()
    # recipients are grouped by the first two words of their cleaned name, e.g. "billa dankt"
    out["cluster"] = out["merchant"].str.split().str[:2].str.join(" ")
    return out


def fixed_category(raw, rules):
    # category of the first fixed rule (config "fixed_categories") whose pattern matches partner and reference
    for rule in rules:
        if re.search(rule["pattern"], raw, re.IGNORECASE):
            return rule["category"]
    return None


def fixed_mask(df):
    # rows matching a fixed rule, e.g. marketplaces whose purchases cannot be told apart by the text
    raw = features(df)["raw"]
    return raw.map(lambda t: fixed_category(t, CONFIG.get("fixed_categories", [])) is not None)


def has_category(df):
    return df["category"].notna() & (df["category"].astype(str).str.strip() != "")


class Classifier:
    def __init__(self, model_file, history):
        self.model = fasttext.load_model(model_file)
        known = history[has_category(history)]
        f = features(known)
        f["category"] = known["category"].astype(str).str.strip()
        # only keys that always had one category are used for the lookup
        self.lookup = {}
        for key in ("text", "iban", "merchant"):
            groups = f[f[key] != ""].groupby(key)["category"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
            self.lookup[key] = groups.dropna().to_dict()
        self.known_clusters = set(f["cluster"]) - {""}
        self.known_ibans = set(f["iban"]) - {""}
        self.household = CONFIG.get("household")

    def is_new_merchant(self, row, feature):
        # the model does not decide for recipients that were never categorized before, they are collected for review
        return bool(feature.cluster) and feature.cluster not in self.known_clusters \
            and feature.iban not in self.known_ibans and not is_member_transfer(row, feature, self.household)

    def model_predictions(self, texts, k=10):
        labels, probabilities = self.model.predict(list(texts), k=k)
        excluded = rule_categories(self.household)
        return [[(l.replace("__label__", ""), float(p)) for l, p in zip(ls, ps) if l.replace("__label__", "") not in excluded]
                for ls, ps in zip(labels, probabilities)]

    def classify(self, df):
        """Return a DataFrame with category, probability and source (lookup, model, rule) for every row of df."""
        f = features(df)
        predictions = self.model_predictions(f["text"])
        result = []
        for (idx, row), feature, prediction in zip(df.iterrows(), f.itertuples(), predictions):
            category, probability, source = None, None, None
            fixed = fixed_category(feature.raw, CONFIG.get("fixed_categories", []))
            if fixed:
                category, probability, source = fixed, 1.0, "fixed"
            # transfers between the household account and its members differ by purpose only, so their IBAN and
            # name must not decide the category
            keys = ("text",) if is_member_transfer(row, feature, self.household) else ("text", "iban", "merchant")
            for key in keys if category is None else ():
                value = getattr(feature, key)
                if value and value in self.lookup[key]:
                    category, probability, source = self.lookup[key][value], 1.0, "lookup"
                    break
            if category is None and self.is_new_merchant(row, feature):
                category, probability, source = "", 0.0, "new merchant"
            elif category is None:
                category, probability = prediction[0] if prediction else ("", 0.0)
                source = "model"
            ruled = household_rule(row, feature, category, prediction, self.household)
            if ruled is not None:
                category, probability, source = ruled
            if probability < MIN_PROBABILITY:
                category = ""
            result.append({"category": category, "probability": round(probability, 3), "source": source})
        return pd.DataFrame(result, index=df.index)


def rule_categories(h):
    # categories that depend on account and counterparty instead of the text, set by the household rules only
    return {h["contribution_category"], h["cash_category"]} if h else set()


def is_member_transfer(row, feature, h):
    # the counterparty (not the reference, which may name the children) is one of the members
    return bool(h) and row["account"] == h["account"] and any(m in feature.merchant for m in h.get("members", []))


def household_rule(row, feature, category, prediction, h):
    # category, probability and source if a household rule applies, else None.
    # prediction: model predictions [(category, probability), ...] without the rule categories
    if not h:
        return None
    contribution, cash = h["contribution_category"], h["cash_category"]
    amount = row["amount.value"]
    withdrawal = amount < 0 and any(k in feature.raw for k in ATM_KEYWORDS)
    if row["account"] == h["account"]:
        member = any(m in feature.merchant for m in h.get("members", []))
        if amount > 0 and member and any(k in feature.ref for k in h.get("contribution_keywords", [])):
            return contribution, 1.0, "rule"
        if amount > 0 and category == contribution:
            # other payments into the household account (top ups, extra contributions)
            return None
        if withdrawal:
            return cash, 1.0, "rule"
        if category in (contribution, cash) and prediction:
            # reimbursements and payments of the household are booked with their real purpose
            label, probability = prediction[0]
            return label, probability, "rule"
        return None
    if feature.iban and feature.iban in {i.replace(" ", "").upper() for i in h.get("ibans", [])}:
        return (cash if amount > 0 else contribution), 1.0, "rule"
    if withdrawal:
        return cash, 1.0, "rule"
    return None
