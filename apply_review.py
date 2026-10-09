"""Apply the decisions of category_audit.xlsx to the history file.

usage: python apply_review.py [category_audit.xlsx] [--history FILE]

- Renames sheet: every category "from" is renamed to "to" (rows without "to" are skipped)
- Review sheet, rows are matched by id:
  - rows above a marker row (decision "hier weiter" or "stop") count as reviewed: the suggested category is used
    (edit the "suggested" column to correct it), a category in "decision" overrides it, "-" keeps the current one
  - rows below the marker, or all rows without a marker: decision "x" takes the suggestion, any other text is used
    as category, empty keeps the current one
- New merchants sheet: "category" is set for all uncategorized transactions of that recipient; empty takes the
  suggestion (if there is one), "-" skips the recipient
A timestamped backup of the history is written first.
"""
import argparse
import shutil
from datetime import datetime

import pandas as pd

from classifier import features, has_category, is_member_transfer
from config import CONFIG, path

parser = argparse.ArgumentParser()
parser.add_argument("review", nargs="?", default=path("category_audit.xlsx"))
parser.add_argument("--history", default=path(CONFIG["history_file"]))
args = parser.parse_args()

sheets = pd.read_excel(args.review, sheet_name=None, dtype=str)
hist = pd.read_excel(args.history, sheet_name="Sheet1")
before = hist["category"].copy()

renames = sheets["Renames"].fillna("")
renames = renames[renames["to"].str.strip() != ""]
for _, r in renames.iterrows():
    hist.loc[hist["category"] == r["from"], "category"] = r["to"].strip()

MARKERS = ("hier weiter", "stop")
review = sheets["Review"].fillna("").reset_index(drop=True)
marker = review.index[review["decision"].str.strip().str.lower().str.startswith(MARKERS)]
reviewed_until = marker[0] if len(marker) else 0
decided = {}
for i, r in review.iterrows():
    decision = r["decision"].strip()
    if i == reviewed_until and len(marker) or decision == "-":
        continue
    if decision.lower() == "x" or (decision == "" and i < reviewed_until):
        category = r["suggested"].strip()
    else:
        category = decision
    if category:
        decided[int(r["id"])] = category
    elif decision.lower() == "x":
        raise SystemExit(f"id {r['id']}: decision 'x' but no suggestion")
if len(marker):
    print(f"rows above the marker (row {reviewed_until + 2} in Excel) are taken as reviewed")
unknown = set(decided) - set(hist["id"])
if unknown:
    raise SystemExit(f"ids not found in {args.history}: {sorted(unknown)[:10]}")
mask = hist["id"].isin(decided)
hist.loc[mask, "category"] = hist.loc[mask, "id"].map(decided)

# new merchants: the category is set for all uncategorized transactions of that recipient
merchants = sheets.get("New merchants", pd.DataFrame(columns=["merchant", "suggested", "category"])).fillna("")
open_rows = ~has_category(hist)
f = features(hist)
member = pd.Series([is_member_transfer(r, ft, CONFIG.get("household")) for (_, r), ft in zip(hist.iterrows(), f.itertuples())],
                   index=hist.index)
merchant_count = 0
for _, m in merchants.iterrows():
    # an empty category takes the suggestion, "-" skips the recipient
    entry = m["category"].strip()
    if entry == "-":
        continue
    category = m["suggested"].strip() if entry.lower() in ("", "x") else entry
    if not category:
        continue
    rows = open_rows & ~member & (f["cluster"] == m["merchant"])
    hist.loc[rows, "category"] = category
    hist.loc[rows, "source"] = "review"
    merchant_count += rows.sum()
hist.loc[mask, "source"] = "review"
hist["transfer"] = hist["category"].isin(CONFIG.get("transfer_categories", []))

changed = (hist["category"].fillna("") != before.fillna("")).sum()
backup = args.history.replace(".xlsx", datetime.now().strftime("_%Y%m%d-%H%M%S.backup.xlsx"))
shutil.copy2(args.history, backup)
hist.to_excel(args.history, sheet_name="Sheet1", index=False)
print(f"{len(renames)} renames, {len(decided)} review decisions, {merchant_count} transactions of new merchants, "
      f"{changed} transactions changed")
print(f"backup written to {backup}, {args.history} updated")
