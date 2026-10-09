"""Train the fastText model on all categorized transactions of the history file.

usage: python train.py [history.xlsx] [--autotune SECONDS]

20 % of every category (with at least 5 entries) is held back to measure the accuracy, then the final model is
trained on all data. Model, vocabulary and metrics are written to /model.
"""
import argparse
import csv
import json
import os
from datetime import datetime

import fasttext
import numpy as np
import pandas as pd

from classifier import rule_categories
from config import CONFIG, path
from sharedfunctions import prep_fasttext

PARAMS = dict(epoch=50, lr=0.5, minn=3, maxn=5)
MIN_VALIDATION_SIZE = 5
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("history", nargs="?", default=path(CONFIG["history_file"]))
parser.add_argument("--autotune", type=int, metavar="SECONDS", help="search parameters with fastText autotune")
args = parser.parse_args()

modeldir = path("model")
os.makedirs(modeldir, exist_ok=True)
train_file, validation_file = os.path.join(modeldir, "train.txt"), os.path.join(modeldir, "validation.txt")
metrics_file = os.path.join(modeldir, "bs.metrics.json")

hist_df = pd.read_excel(args.history, sheet_name="Sheet1")
prep_df = hist_df[hist_df["category"].notna() & (hist_df["category"].astype(str).str.strip() != "")].copy()
prep_df["category"] = prep_df["category"].astype(str).str.strip()
prep_df = prep_fasttext(prep_df)
# entries without any text besides the label carry no information
prep_df = prep_df[prep_df["fasttext"].str.split().str.len() > 1]
# categories set by the household rules depend on account and counterparty, not on the text
prep_df = prep_df[~prep_df["category"].isin(rule_categories(CONFIG.get("household")))]

# stratified split, small categories are used for training only
rng = np.random.default_rng(SEED)
validation_idx = []
for category, group in prep_df.groupby("category"):
    if len(group) >= MIN_VALIDATION_SIZE:
        validation_idx += list(rng.choice(group.index, size=round(len(group) * 0.2), replace=False))
validation_df = prep_df.loc[validation_idx]
train_df = prep_df.drop(index=validation_idx)


def write(df, filename):
    df["fasttext"].to_csv(filename, index=False, header=False, quoting=csv.QUOTE_NONE, escapechar="\\")


def train(filename, **params):
    # training can diverge (NaN) with a too high learning rate, retry with a lower one
    params = {**params, "seed": SEED, "verbose": 0}
    for _ in range(4):
        try:
            return fasttext.train_supervised(filename, **params), params
        except RuntimeError as error:
            if "NaN" not in str(error):
                raise
            params["lr"] = params.get("lr", 0.1) / 2
            print(f"training diverged, retrying with lr={params['lr']}")
    raise RuntimeError("training diverged repeatedly")


write(train_df, train_file)
write(validation_df, validation_file)
if args.autotune:
    model = fasttext.train_supervised(train_file, autotuneValidationFile=validation_file,
                                      autotuneDuration=args.autotune, verbose=0)
    params = {name: getattr(model.f.getArgs(), name) for name in ("epoch", "lr", "minn", "maxn", "wordNgrams", "dim")}
    print(f"autotune found {params}")
else:
    model, params = train(train_file, **PARAMS)

n, precision, _ = model.test(validation_file)
print(f"{len(prep_df)} categorized transactions, {prep_df['category'].nunique()} categories")
print(f"validation on {n} transactions: P@1 {precision:.3f}")

# per category recall on the validation set to show weak categories
predicted, _ = model.predict(validation_df["fasttext"].str.replace(r"^__label__\S+\s*", "", regex=True).tolist(), k=1)
validation_df = validation_df.assign(predicted=[p[0].replace("__label__", "") for p in predicted])
recall = validation_df.assign(hit=validation_df["category"] == validation_df["predicted"]).groupby("category")["hit"].agg(["size", "mean"])
weak = recall[recall["mean"] < 0.8].sort_values("mean")
if not weak.empty:
    print("categories with recall below 80 %:")
    print(weak.rename(columns={"size": "validated", "mean": "recall"}).round(2).to_string())
counts = prep_df["category"].value_counts()
rare = counts[counts < MIN_VALIDATION_SIZE]
if not rare.empty:
    print(f"categories with less than {MIN_VALIDATION_SIZE} entries (not validated, hardly ever predicted): "
          + ", ".join(f"{c} ({n})" for c, n in rare.items()))

if os.path.exists(metrics_file):
    with open(metrics_file, encoding="utf-8") as f:
        previous = json.load(f)
    print(f"previous model: P@1 {previous['precision']:.3f} on {previous['validated']} transactions ({previous['trained']})")

# final model on all data
write(prep_df, train_file)
model, params = train(train_file, **{k: v for k, v in params.items() if k not in ("seed", "verbose")})
model.save_model(os.path.join(modeldir, "bs.model"))
with open(os.path.join(modeldir, "bs.words"), 'w', encoding="utf-8") as f:
    print(model.labels, file=f)
    print(model.words, file=f)
with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump({"trained": datetime.now().isoformat(timespec="seconds"), "history": os.path.basename(args.history),
               "transactions": len(prep_df), "validated": n, "precision": round(precision, 4),
               "params": {k: v for k, v in params.items() if k != "verbose"}}, f, indent=2)
print(f"model written to {modeldir}")
