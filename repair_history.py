"""Check the history file against the bank exports in /input and write a repaired copy.

For every account and day covered by an export (empty exports count as covered without transactions),
transactions are compared by (account, booking, amount):
- transactions missing in the history are added (without category, import.py classifies them on its next run)
- surplus history rows, where the export holds fewer identical transactions, are removed as double imports
- history rows without any counterpart in the export are moved to the legacy account configured in
  config.json ("legacy_accounts"), e.g. a closed account that was imported under the same name, or flagged
Rows that could not be verified against an export are flagged if they look like double imports.
Text garbled by reading utf-8 files as latin-1 is repaired. The history file itself is never modified.
"""
import os
import re

import pandas as pd

from config import CONFIG, path
from importers import KEY, combine_statements, ensure_ids, new_transactions, read_statement, similarity, text_of

history_filename = path(CONFIG["history_file"])
repaired_filename = history_filename.replace(".xlsx", "_repaired.xlsx")
inputdir = path("input")
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


def covered_days(periods):
    days = set()
    for start, end in periods:
        days.update(pd.date_range(start, end).strftime("%Y-%m-%d"))
    return days


hist = pd.read_excel(history_filename, sheet_name="Sheet1")
hist["booking"] = pd.to_datetime(hist["booking"].astype(str).str[:10]).dt.strftime("%Y-%m-%d")
hist["amount.value"] = hist["amount.value"].round(2)
for column in TEXT_COLUMNS:
    if column in hist.columns:
        hist[column] = hist[column].map(fix_mojibake)
hist["repair"] = ""

# exports per account, with the periods they cover
exports, periods = {}, {}
for filename in sorted(os.listdir(inputdir)):
    df = read_statement(os.path.join(inputdir, filename))
    if df is None or df.attrs["account"] is None:
        continue
    account = df.attrs["account"]
    exports.setdefault(account, []).append(df)
    if df.attrs["period"]:
        periods.setdefault(account, []).append(df.attrs["period"])

added, removed, verified = [], [], set()
for account, frames in exports.items():
    export = combine_statements(frames)
    last_history_day = hist.loc[hist["account"] == account, "booking"].max()
    days = {d for d in covered_days(periods.get(account, [])) if d <= last_history_day}
    in_range = hist[(hist["account"] == account) & hist["booking"].isin(days)]
    export = export[export["booking"].isin(days)]
    print(f"{account}: comparing {len(in_range)} history rows with {len(export)} exported transactions")

    added.append(new_transactions(in_range, export).assign(repair="added: missing in history"))

    export_groups = {k: g for k, g in export.groupby(KEY)}
    legacy = CONFIG.get("legacy_accounts", {}).get(account)
    for k, group in in_range.groupby(KEY):
        exp = export_groups.get(k)
        if exp is None:
            if legacy:
                hist.loc[group.index, "account"] = legacy
                hist.loc[group.index, "repair"] = f"moved: not in export of {account}"
            else:
                hist.loc[group.index, "repair"] = "flag: not in export"
            continue
        verified.update(group.index)
        if len(group) <= len(exp):
            continue
        # keep the history rows that match the exported transactions best, the rest are double imports
        keep = set()
        for _, e in exp.iterrows():
            candidates = [(similarity(text_of(e), text_of(h)), i) for i, h in group.iterrows() if i not in keep]
            keep.add(max(candidates)[1])
        removed.extend(i for i in group.index if i not in keep)

hist.loc[removed, "repair"] = "removed: double import"
for idx in probable_doubles(hist.drop(index=list(verified))):
    hist.loc[idx, "repair"] = (hist.loc[idx, "repair"] + "; " if hist.loc[idx, "repair"] else "") + "flag: probable double import"

added = pd.concat(added) if added else pd.DataFrame()
log = pd.concat([hist[hist["repair"] != ""], added]).sort_values(["repair", "account", "booking"])
repaired = pd.concat([hist.drop(index=removed), added], ignore_index=True)
repaired.sort_values(["booking", "account", "amount.value", "reference"], inplace=True)
repaired = ensure_ids(repaired)

with pd.ExcelWriter(repaired_filename) as writer:
    repaired.to_excel(writer, sheet_name="Sheet1", index=False)
    log.to_excel(writer, sheet_name="Repair log", index=False)

print(log.groupby(["account", "repair"]).agg(rows=("amount.value", "size"), amount=("amount.value", "sum")).round(2))
print(f"{len(hist)} -> {len(repaired)} transactions, written to {repaired_filename}")
