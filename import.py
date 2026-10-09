import os
import shutil
from datetime import datetime

import fasttext
import pandas as pd

from config import CONFIG, path
from importers import FIELDS, combine_statements, new_transactions, read_statement
from sharedfunctions import prep_fasttext

# specify filename which holds complete history of transaction data
history_filename = path(CONFIG["history_file"])
new_import_filename = path("new_import.xlsx")
inputdir = path("input")
modelfile = path(os.path.join("model", "bs.model"))

# read all supported files from /input, overlapping exports are combined without double counting
frames = []
for filename in sorted(os.listdir(inputdir)):
    df = read_statement(os.path.join(inputdir, filename))
    if df is None or df.empty:
        print(f"skipped {filename}")
        continue
    print(f"read {len(df):5} transactions from {filename}")
    frames.append(df)
if not frames:
    raise SystemExit("no suitable files in /input")
input_df = combine_statements(frames)

# only keep transactions that are not yet part of the history
if os.path.exists(history_filename):
    hist_df = pd.read_excel(history_filename, sheet_name="Sheet1")
    hist_df["booking"] = pd.to_datetime(hist_df["booking"].astype(str).str[:10]).dt.strftime("%Y-%m-%d")
    hist_df["amount.value"] = hist_df["amount.value"].round(2)
else:
    hist_df = pd.DataFrame(columns=FIELDS + ["category"])
input_df = new_transactions(hist_df, input_df).reset_index(drop=True)
print(f"{len(input_df)} new transactions")

# classify new transactions and history entries which are still missing a category
model = fasttext.load_model(modelfile)


def classify(df):
    texts = prep_fasttext(df[FIELDS].copy())["fasttext"]
    labels, probabilities = model.predict(texts.tolist(), k=1)
    categories = [l[0].replace("__label__", "") if p[0] >= 0.5 else "" for l, p in zip(labels, probabilities)]
    return categories, [round(float(p[0]), 3) for p in probabilities]


if not input_df.empty:
    input_df["category"], input_df["probability"] = classify(input_df)
    print(input_df[["booking", "account", "amount.value", "category"]])
    input_df.to_excel(new_import_filename, sheet_name="Sheet1", index=False)

uncategorized = hist_df["category"].isna() | (hist_df["category"].astype(str).str.strip() == "")
if uncategorized.any():
    hist_df.loc[uncategorized, "category"], hist_df.loc[uncategorized, "probability"] = classify(hist_df[uncategorized])
    found = (hist_df.loc[uncategorized, "category"] != "").sum()
    print(f"{found} of {uncategorized.sum()} uncategorized history entries got a category")

# add new transactions to existing historical DB, keep a backup of the previous version
if os.path.exists(history_filename):
    backup = history_filename.replace(".xlsx", datetime.now().strftime("_%Y%m%d-%H%M%S.backup.xlsx"))
    shutil.copy2(history_filename, backup)
    print(f"backup written to {backup}")
df = pd.concat([hist_df, input_df], ignore_index=True)
# mark transfers between own accounts, so they can be excluded from income and spending
df["transfer"] = df["category"].isin(CONFIG.get("transfer_categories", []))
df.sort_values(["booking", "account", "amount.value", "reference"], inplace=True)
df.to_excel(history_filename, sheet_name="Sheet1", index=False)
print(f"{history_filename} now holds {len(df)} transactions")
