"""List inconsistent categorizations of the history file for manual review.

usage: python audit_categories.py [history.xlsx] [--categories CAT ...]

Writes category_audit.xlsx with the sheets
- Review: one row per transaction to check, with the current and the suggested category and the reason.
  Fill the "decision" column: x = take the suggestion, a category name = use that one, empty = keep as is.
- New merchants: uncategorized transactions of recipients never seen before, one row per recipient. Fill "category"
  (empty = take the suggestion, "-" = skip); it is set for all their transactions and later imports recognize the
  recipient.
- Renames: category renames applied to all transactions (fill "to", empty = keep)
- Overview: every category with count, period, money in/out and whether it counts as transfer
- Category names: near identical category names and rarely used categories
- Merchants: counterparties booked to more than one category
Apply the decisions with apply_review.py.

Reasons for review:
- household rule: booking contradicts the household rules in config.json
- same text: an identical text is mostly booked to another category
- model: a model trained without this transaction confidently suggests another category
- review category: all transactions of the categories given with --categories
- uncategorized: transactions the classifier was not sure about
"""
import argparse
import csv
import difflib
import os
import tempfile

import fasttext
import numpy as np
import pandas as pd
from unidecode import unidecode

from classifier import (MIN_PROBABILITY, features, fixed_mask, has_category, household_rule, is_member_transfer,
                        rule_categories)
from config import CONFIG, path
from sharedfunctions import MODEL_PARAMS

parser = argparse.ArgumentParser()
parser.add_argument("history", nargs="?", default=path(CONFIG["history_file"]))
parser.add_argument("--categories", nargs="*", default=[], help="list all transactions of these categories for review")
args = parser.parse_args()

COLUMNS = ["id", "booking", "account", "amount.value", "partnerName", "reference", "category"]
RARE = 5
FOLDS = 5
MIN_CONFIDENCE = 0.8

hist = pd.read_excel(args.history, sheet_name="Sheet1")
if "id" not in hist.columns:
    raise SystemExit("history has no id column, run repair_history.py or import.py first")
# transactions with a fixed category (config "fixed_categories") keep their manual categories and are not audited
df = hist[has_category(hist) & ~fixed_mask(hist)].copy()
df["category"] = df["category"].astype(str).str.strip()
df["booking"] = df["booking"].astype(str).str[:10]
f = features(df)
df["text"] = f["text"]
df["merchant"] = f["merchant"].str.split().str[:3].str.join(" ")
counts = df["category"].value_counts()

# overview of all categories
transfer = set(CONFIG.get("transfer_categories", []))
overview = df.groupby("category").agg(
    count=("amount.value", "size"), first=("booking", "min"), last=("booking", "max"),
    money_in=("amount.value", lambda a: round(a[a > 0].sum(), 2)), money_out=("amount.value", lambda a: round(a[a < 0].sum(), 2)),
    accounts=("account", lambda a: ", ".join(sorted(a.astype(str).unique()))))
overview["transfer"] = overview.index.isin(transfer)
overview = overview.sort_values("count", ascending=False).reset_index()

# category names that look like variants of each other, and rare categories
names, renames = [], []
cats = list(counts.index)
for i, a in enumerate(cats):
    for b in cats[i + 1:]:
        la, lb = a.lower(), b.lower()
        ratio = difflib.SequenceMatcher(None, la, lb).ratio()
        if la == lb or ratio >= 0.85 or (len(la) > 3 and len(lb) > 3 and (la.startswith(lb) or lb.startswith(la))):
            names.append({"issue": "similar names", "category": a, "count": counts[a], "other": b, "other_count": counts[b],
                          "similarity": round(ratio, 2)})
            # spelling variants (case, lost umlauts) are renamed to the more frequent spelling right away
            variant = la == lb or unidecode(la).replace("ae", "a").replace("oe", "o").replace("ue", "u") == \
                unidecode(lb).replace("ae", "a").replace("oe", "o").replace("ue", "u") or \
                difflib.SequenceMatcher(None, unidecode(la), unidecode(lb)).ratio() >= 0.9
            major, minor = (a, b) if counts[a] >= counts[b] else (b, a)
            renames.append({"from": minor, "from_count": counts[minor], "to": major if variant else "",
                            "candidate": major})
for c, n in counts[counts < RARE].items():
    names.append({"issue": f"rare (< {RARE} entries)", "category": c, "count": n})
names = pd.DataFrame(names)
renames = pd.DataFrame(renames, columns=["from", "from_count", "to", "candidate"])

# counterparties with several categories
merchants = []
for merchant, g in df[df["merchant"] != ""].groupby("merchant"):
    dist = g["category"].value_counts()
    if len(dist) > 1:
        merchants.append({"merchant": merchant, "transactions": len(g), "majority": dist.index[0],
                          "majority_share": round(dist.iloc[0] / len(g), 2),
                          "other_categories": ", ".join(f"{c} ({n})" for c, n in dist.iloc[1:].items()),
                          "example": g["partnerName"].astype(str).iloc[0]})
merchants = pd.DataFrame(merchants).sort_values(["majority_share", "transactions"], ascending=[True, False])

# out of fold predictions: every transaction is predicted by a model that did not see it during training
# categories set by the household rules depend on account and counterparty, not on the text, so the model
# does not learn them
household = CONFIG.get("household")
RULE_CATEGORIES = rule_categories(household)
rng = np.random.default_rng(42)
df["fold"] = rng.integers(0, FOLDS, len(df))
predictions = {}
with tempfile.TemporaryDirectory() as tmp:
    train_file = os.path.join(tmp, "train.txt")
    for fold in range(FOLDS):
        train = df[(df["fold"] != fold) & (df["text"] != "") & ~df["category"].isin(RULE_CATEGORIES)]
        ("__label__" + train["category"] + " " + train["text"]).to_csv(train_file, index=False, header=False,
                                                                       quoting=csv.QUOTE_NONE, escapechar="\\")
        model = fasttext.train_supervised(train_file, **MODEL_PARAMS, seed=42, verbose=0)
        test = df["fold"] == fold
        labels, probs = model.predict(df.loc[test, "text"].tolist(), k=10)
        for idx, ls, ps in zip(df.index[test], labels, probs):
            predictions[idx] = [(l.replace("__label__", ""), float(p)) for l, p in zip(ls, ps)]
predictions = pd.Series(predictions)
df["predicted"] = predictions[df.index].map(lambda p: p[0][0])
df["confidence"] = predictions[df.index].map(lambda p: round(p[0][1], 3))
accuracy = (df["predicted"] == df["category"])[~df["category"].isin(RULE_CATEGORIES)].mean()

# transactions to review, the first reason found per transaction determines the suggestion.
# clear cases are preselected with decision "x": rule based corrections, purposes the model is sure about and
# texts that are booked consistently elsewhere
PRESELECT_CONFIDENCE = 0.8
PRESELECT_SHARE = 0.75
review = {}


def add(row, suggested, reason, confidence=None, preselect=False):
    if row.name in review:
        review[row.name]["reason"] += f"; {reason}"
        return
    review[row.name] = {**row[COLUMNS].to_dict(), "suggested": suggested, "confidence": confidence, "reason": reason,
                        "decision": "x" if preselect and suggested else ""}


for (idx, row), feature in zip(df.iterrows(), f.loc[df.index].itertuples()):
    ruled = household_rule(row, feature, row["category"], predictions[idx], household)
    if ruled and ruled[0] != row["category"]:
        category, probability, _ = ruled
        if row["account"] != household["account"]:
            # transfers with the household account are unambiguous, cash withdrawals booked with a purpose are kept
            via_household = feature.iban in {i.replace(" ", "").upper() for i in household.get("ibans", [])}
            add(row, category, "household rule: own account" + ("" if via_household else " cash withdrawal"), 1.0,
                preselect=via_household)
        elif category == household["contribution_category"]:
            add(row, category, "household rule: contribution", 1.0, preselect=True)
        else:
            add(row, category, "household rule: real purpose", round(probability, 3),
                preselect=probability >= PRESELECT_CONFIDENCE)

for text, g in df[df["text"] != ""].groupby("text"):
    dist = g["category"].value_counts()
    if len(dist) > 1:
        share = dist.iloc[0] / len(g)
        for _, row in g[g["category"] != dist.index[0]].iterrows():
            add(row, dist.index[0], f"same text: {', '.join(f'{c} ({n})' for c, n in dist.items())}", round(share, 2),
                preselect=share >= PRESELECT_SHARE and dist.iloc[0] >= 3)

# the model alone is overconfident on one-off merchants, so a disagreement is only listed when the other
# transactions with the same counterparty are mostly booked like the model suggests
disagree = df[(df["predicted"] != df["category"]) & (df["confidence"] >= MIN_CONFIDENCE) & ~df["category"].isin(RULE_CATEGORIES)]
for _, row in disagree.sort_values("confidence", ascending=False).iterrows():
    others = df[(df["merchant"] == row["merchant"]) & (df.index != row.name)]["category"].value_counts()
    if row["merchant"] and len(others) and others.index[0] == row["predicted"] and others.iloc[0] >= 2:
        add(row, row["predicted"], f"model: {row['confidence']:.2f}, counterparty mostly {row['predicted']} "
            f"({others.iloc[0]} of {others.sum()})", row["confidence"])

for _, row in df[df["category"].isin(args.categories)].iterrows():
    add(row, row["predicted"] if row["predicted"] != row["category"] else "", f"review category {row['category']}",
        row["confidence"])

# transactions without category: recipients never seen before are grouped per recipient on the sheet "New merchants",
# the others (transfers with the household members, unclear purposes) are listed on the review sheet
open_rows = hist[~has_category(hist) & ~fixed_mask(hist)].copy()
new_merchants = pd.DataFrame(columns=["merchant", "transactions", "total", "first", "last", "accounts", "examples",
                                      "suggested", "confidence", "category"])
if not open_rows.empty:
    open_rows["booking"] = open_rows["booking"].astype(str).str[:10]
    open_rows["category"] = ""
    of = features(open_rows)
    with tempfile.TemporaryDirectory() as tmp:
        train_file = os.path.join(tmp, "train.txt")
        train = df[(df["text"] != "") & ~df["category"].isin(RULE_CATEGORIES)]
        ("__label__" + train["category"] + " " + train["text"]).to_csv(train_file, index=False, header=False,
                                                                       quoting=csv.QUOTE_NONE, escapechar="\\")
        model = fasttext.train_supervised(train_file, **MODEL_PARAMS, seed=42, verbose=0)
        labels, probs = model.predict(of["text"].tolist(), k=1)
    open_rows["suggested"] = [l[0].replace("__label__", "") for l in labels]
    open_rows["confidence"] = [round(float(p[0]), 3) for p in probs]
    member = pd.Series([is_member_transfer(r, ft, household) for (_, r), ft in zip(open_rows.iterrows(), of.itertuples())],
                       index=open_rows.index)
    grouped = (of["cluster"] != "") & ~member
    clusters = []
    for cluster, g in open_rows[grouped].groupby(of.loc[grouped, "cluster"]):
        best = g.sort_values("confidence", ascending=False).iloc[0]
        refs = (g["partnerName"].fillna("").astype(str) + " | " + g["reference"].fillna("").astype(str)).str.replace(r"\s+", " ", regex=True)
        clusters.append({"merchant": cluster, "transactions": len(g), "total": round(g["amount.value"].sum(), 2),
                         "first": g["booking"].min(), "last": g["booking"].max(),
                         "accounts": ", ".join(sorted(g["account"].astype(str).unique())),
                         "examples": " || ".join(refs.str[:60].unique()[:3]),
                         "suggested": best["suggested"] if best["confidence"] >= MIN_PROBABILITY else "",
                         "confidence": best["confidence"], "category": ""})
    if clusters:
        new_merchants = pd.DataFrame(clusters).sort_values(["transactions", "total"], ascending=[False, True])
    for _, row in open_rows[~grouped].iterrows():
        review[("open", row.name)] = {**row[COLUMNS].to_dict(),
                                      "suggested": row["suggested"] if row["confidence"] >= MIN_PROBABILITY else "",
                                      "confidence": row["confidence"], "reason": "uncategorized", "decision": ""}

review = pd.DataFrame(review.values(), columns=COLUMNS + ["suggested", "confidence", "reason", "decision"])
review["sort_partner"] = review["partnerName"].fillna("").astype(str).str.lower()
review = review.sort_values(["decision", "reason", "category", "sort_partner", "booking"], ascending=[False, True, True, True, True])
review = review.drop(columns="sort_partner")

output = path("category_audit.xlsx")
with pd.ExcelWriter(output) as writer:
    review.to_excel(writer, sheet_name="Review", index=False)
    new_merchants.to_excel(writer, sheet_name="New merchants", index=False)
    renames.to_excel(writer, sheet_name="Renames", index=False)
    overview.to_excel(writer, sheet_name="Overview", index=False)
    names.to_excel(writer, sheet_name="Category names", index=False)
    merchants.to_excel(writer, sheet_name="Merchants", index=False)

print(f"{len(df)} categorized transactions in {len(counts)} categories, cross validated accuracy {accuracy:.3f}")
print(f"similar category names: {len(renames)} ({(renames['to'] != '').sum()} spelling variants preselected), "
      f"rare categories: {(counts < RARE).sum()}, counterparties with several categories: {len(merchants)}")
print("transactions to review by reason:")
print(review.assign(kind=review["reason"].str.split(";").str[0].str.split(":").str[0], preselected=review["decision"] == "x")
      .groupby(["kind", "preselected"]).size().unstack(fill_value=0).to_string())
print(f"new merchants: {len(new_merchants)} recipients with {int(new_merchants['transactions'].sum()) if len(new_merchants) else 0} transactions")
print(f"written to {output}")
