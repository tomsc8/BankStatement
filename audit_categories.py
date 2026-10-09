"""List inconsistent categorizations of the history file for manual review.

usage: python audit_categories.py [history.xlsx]

Writes category_audit.xlsx with the sheets
- Overview: every category with count, period, money in/out and whether it counts as transfer
- Category names: near identical category names and rarely used categories
- Same text: transactions with identical text but different categories
- Merchants: counterparties booked to more than one category
- Model disagreement: transactions a model trained without them confidently puts into another category
Each sheet has an empty "decision" column for your choice.
"""
import argparse
import csv
import difflib
import os
import tempfile

import fasttext
import numpy as np
import pandas as pd

from config import CONFIG, path
from sharedfunctions import clean_text

parser = argparse.ArgumentParser()
parser.add_argument("history", nargs="?", default=path(CONFIG["history_file"]))
args = parser.parse_args()

COLUMNS = ["booking", "account", "amount.value", "partnerName", "reference", "category"]
RARE = 5
FOLDS = 5
MIN_CONFIDENCE = 0.8

hist = pd.read_excel(args.history, sheet_name="Sheet1")
df = hist[hist["category"].notna() & (hist["category"].astype(str).str.strip() != "")].copy()
df["category"] = df["category"].astype(str).str.strip()
df["booking"] = df["booking"].astype(str).str[:10]
df["text"] = (df["partnerName"].fillna("").astype(str) + " " + df["reference"].fillna("").astype(str)).map(clean_text)
df["merchant"] = df["partnerName"].fillna("").astype(str).map(clean_text).str.split().str[:3].str.join(" ")
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
names = []
cats = list(counts.index)
for i, a in enumerate(cats):
    for b in cats[i + 1:]:
        la, lb = a.lower(), b.lower()
        ratio = difflib.SequenceMatcher(None, la, lb).ratio()
        if la == lb or ratio >= 0.85 or (len(la) > 3 and len(lb) > 3 and (la.startswith(lb) or lb.startswith(la))):
            names.append({"issue": "similar names", "category": a, "count": counts[a], "other": b, "other_count": counts[b],
                          "similarity": round(ratio, 2)})
for c, n in counts[counts < RARE].items():
    names.append({"issue": f"rare (< {RARE} entries)", "category": c, "count": n})
names = pd.DataFrame(names)

# identical texts with different categories
groups = df[df["text"] != ""].groupby("text")
same_text = []
for text, g in groups:
    if g["category"].nunique() > 1:
        dist = g["category"].value_counts()
        for _, row in g.iterrows():
            same_text.append({"text": text, "categories": ", ".join(f"{c} ({n})" for c, n in dist.items()),
                              "majority": dist.index[0], **row[COLUMNS].to_dict()})
same_text = pd.DataFrame(same_text)
if not same_text.empty:
    same_text = same_text[same_text["category"] != same_text["majority"]].sort_values(["text", "booking"])

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
rng = np.random.default_rng(42)
usable = df[df["text"].str.split().str.len() > 0].copy()
usable["fold"] = rng.integers(0, FOLDS, len(usable))
usable["predicted"], usable["confidence"] = "", 0.0
with tempfile.TemporaryDirectory() as tmp:
    train_file = os.path.join(tmp, "train.txt")
    for fold in range(FOLDS):
        train = usable[usable["fold"] != fold]
        ("__label__" + train["category"] + " " + train["text"]).to_csv(train_file, index=False, header=False,
                                                                       quoting=csv.QUOTE_NONE, escapechar="\\")
        model = fasttext.train_supervised(train_file, epoch=50, lr=0.5, minn=3, maxn=5, seed=42, verbose=0)
        test = usable["fold"] == fold
        labels, probs = model.predict(usable.loc[test, "text"].tolist(), k=1)
        usable.loc[test, "predicted"] = [l[0].replace("__label__", "") for l in labels]
        usable.loc[test, "confidence"] = [round(float(p[0]), 3) for p in probs]
disagree = usable[(usable["predicted"] != usable["category"]) & (usable["confidence"] >= MIN_CONFIDENCE)]
disagree = disagree[COLUMNS + ["predicted", "confidence"]].sort_values("confidence", ascending=False)
accuracy = (usable["predicted"] == usable["category"]).mean()

output = path("category_audit.xlsx")
with pd.ExcelWriter(output) as writer:
    for name, sheet in [("Overview", overview), ("Category names", names), ("Same text", same_text),
                        ("Merchants", merchants), ("Model disagreement", disagree)]:
        sheet = sheet.copy()
        sheet["decision"] = ""
        sheet.to_excel(writer, sheet_name=name, index=False)

print(f"{len(df)} categorized transactions in {len(counts)} categories, cross validated accuracy {accuracy:.3f}")
print(f"similar category names: {(names['issue'] == 'similar names').sum() if not names.empty else 0}, "
      f"rare categories: {(counts < RARE).sum()}")
print(f"transactions deviating from the majority category of an identical text: {len(same_text)}")
print(f"counterparties with several categories: {len(merchants)}")
print(f"confident model disagreements (>= {MIN_CONFIDENCE}): {len(disagree)}")
print(f"written to {output}")
