"""Apply the decisions of category_audit.xlsx to the history file.

usage: python apply_review.py [category_audit.xlsx] [--history FILE]

- Renames sheet: every category "from" is renamed to "to" (rows without "to" are skipped)
- Review sheet: decision "x" takes the suggested category, any other text is used as category,
  empty keeps the current one. Rows are matched by id.
A timestamped backup of the history is written first.
"""
import argparse
import shutil
from datetime import datetime

import pandas as pd

from config import CONFIG, path

parser = argparse.ArgumentParser()
parser.add_argument("review", nargs="?", default=path("category_audit.xlsx"))
parser.add_argument("--history", default=path(CONFIG["history_file"]))
args = parser.parse_args()

sheets = pd.read_excel(args.review, sheet_name=["Review", "Renames"], dtype=str)
hist = pd.read_excel(args.history, sheet_name="Sheet1")
before = hist["category"].copy()

renames = sheets["Renames"].fillna("")
renames = renames[renames["to"].str.strip() != ""]
for _, r in renames.iterrows():
    hist.loc[hist["category"] == r["from"], "category"] = r["to"].strip()

review = sheets["Review"].fillna("")
review = review[review["decision"].str.strip() != ""]
decided = {}
for _, r in review.iterrows():
    decision = r["decision"].strip()
    category = r["suggested"] if decision.lower() == "x" else decision
    if not category:
        raise SystemExit(f"id {r['id']}: decision 'x' but no suggestion")
    decided[int(r["id"])] = category
unknown = set(decided) - set(hist["id"])
if unknown:
    raise SystemExit(f"ids not found in {args.history}: {sorted(unknown)[:10]}")
mask = hist["id"].isin(decided)
hist.loc[mask, "category"] = hist.loc[mask, "id"].map(decided)
hist.loc[mask, "source"] = "review"
hist["transfer"] = hist["category"].isin(CONFIG.get("transfer_categories", []))

changed = (hist["category"].fillna("") != before.fillna("")).sum()
backup = args.history.replace(".xlsx", datetime.now().strftime("_%Y%m%d-%H%M%S.backup.xlsx"))
shutil.copy2(args.history, backup)
hist.to_excel(args.history, sheet_name="Sheet1", index=False)
print(f"{len(renames)} renames, {len(decided)} review decisions, {changed} transactions changed")
print(f"backup written to {backup}, {args.history} updated")
