"""Check all_statements.xlsx against the bank exports in /input and write a repaired copy.

For every account and day covered by an export, transactions are compared by (account, booking, amount):
- transactions missing in the history are added (without category, import.py classifies them on its next run)
- surplus history rows, where the export holds fewer identical transactions, are removed as double imports
- history rows without any counterpart in the export are kept and flagged
Outside the covered periods, rows that look like double imports are flagged but kept.
Text garbled by reading utf-8 files as latin-1 is repaired. all_statements.xlsx itself is never modified.
"""
import os
import re

import pandas as pd

from importers import KEY, combine_statements, read_statement, similarity, text_of, new_transactions

basedir = os.path.dirname(os.path.abspath(__file__))
history_filename = os.path.join(basedir, "all_statements.xlsx")
repaired_filename = os.path.join(basedir, "all_statements_repaired.xlsx")
inputdir = os.path.join(basedir, "input")
TEXT_COLUMNS = ["partnerName", "reference", "fasttext"]


def fix_mojibake(value):
    if not isinstance(value, str) or not re.search("[ÃÂâ]", value):
        return value
    for encoding in ("cp1252", "latin_1"):
        try:
            return value.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return value


def reference_core(value):
    # reference without the suffixes that differ between export formats, used to spot double imports
    value = str(value) if pd.notna(value) else ""
    value = re.split(r"AWV-MELDEPFLICHT|DATUM \d\d\.\d\d\.\d{4}", value)[0]
    value = re.sub(r"\s+", "", value).lower()
    if re.search(r"debitk\.\d+visadebit|visadebitkartenumsatz", value):
        value = "visacard"
    return value


def partner_core(value):
    return re.sub(r"[^a-z]", "", str(value).lower())[:5] if pd.notna(value) else ""


def probable_doubles(df):
    # index of rows which repeat another row of the same day and amount with the same partner and reference
    doubles = []
    for _, group in df[df["amount.value"] != 0].groupby(KEY):
        if len(group) < 2:
            continue
        refs = group["reference"].map(reference_core)
        partners = group["partnerName"].map(partner_core)
        seen = []
        for idx in group.index:
            r, p = refs[idx], partners[idx]
            if any((r == r2 or (r and r2 and (r.startswith(r2) or r2.startswith(r))))
                   and (p == p2 or "" in (p, p2)) for r2, p2 in seen):
                doubles.append(idx)
            else:
                seen.append((r, p))
    return doubles


hist = pd.read_excel(history_filename, sheet_name="Sheet1")
hist["booking"] = pd.to_datetime(hist["booking"].astype(str).str[:10]).dt.strftime("%Y-%m-%d")
hist["amount.value"] = hist["amount.value"].round(2)
for column in TEXT_COLUMNS:
    if column in hist.columns:
        hist[column] = hist[column].map(fix_mojibake)
hist["repair"] = ""

# exports per account, with the days each export covers
exports, covered = {}, {}
for filename in sorted(os.listdir(inputdir)):
    df = read_statement(os.path.join(inputdir, filename))
    if df is None:
        continue
    account = df["account"].iloc[0]
    exports.setdefault(account, []).append(df)
    days = pd.date_range(df["booking"].min(), df["booking"].max()).strftime("%Y-%m-%d")
    covered.setdefault(account, set()).update(days)

added, removed = [], []
for account, frames in exports.items():
    export = combine_statements(frames)
    in_range = hist[(hist["account"] == account) & hist["booking"].isin(covered[account])]
    export = export[export["booking"] <= hist.loc[hist["account"] == account, "booking"].max()]
    print(f"{account}: comparing {len(in_range)} history rows with {len(export)} exported transactions")

    missing = new_transactions(in_range, export)
    added.append(missing.assign(repair="added: missing in history"))

    export_groups = {k: g for k, g in export.groupby(KEY)}
    for k, group in in_range.groupby(KEY):
        exp = export_groups.get(k)
        if exp is None:
            hist.loc[group.index, "repair"] = "flag: not in export"
            continue
        if len(group) <= len(exp):
            continue
        # keep the history rows that match the exported transactions best, the rest are double imports
        keep = set()
        for _, e in exp.iterrows():
            candidates = [(similarity(text_of(e), text_of(h)), i) for i, h in group.iterrows() if i not in keep]
            keep.add(max(candidates)[1])
        removed.extend(i for i in group.index if i not in keep)

hist.loc[removed, "repair"] = "removed: double import"
outside = hist[~hist.apply(lambda r: r["booking"] in covered.get(r["account"], ()), axis=1)]
flagged = probable_doubles(outside)
hist.loc[flagged, "repair"] = "flag: probable double import"

added = pd.concat(added) if added else pd.DataFrame()
log = pd.concat([hist[hist["repair"] != ""], added]).sort_values(["repair", "account", "booking"])
repaired = pd.concat([hist.drop(index=removed), added])
repaired.sort_values(["booking", "account", "amount.value", "reference"], inplace=True)

with pd.ExcelWriter(repaired_filename) as writer:
    repaired.to_excel(writer, sheet_name="Sheet1", index=False)
    log.to_excel(writer, sheet_name="Repair log", index=False)

print(log.groupby(["account", "repair"]).agg(rows=("amount.value", "size"), amount=("amount.value", "sum")).round(2))
print(f"{len(hist)} -> {len(repaired)} transactions, written to {repaired_filename}")
