"""Generate a fully synthetic, randomized demo dataset in the all_statements.xlsx format."""
import random
import sys
from datetime import date, timedelta

import pandas as pd

random.seed(42)

# category -> (fictional partner names, reference templates, amount range, accounts)
CATEGORIES = {
    "Lebensmittel": (["Frischmarkt", "Biohof Sonnental", "Supermarkt Mitte", "Baeckerei Kornfeld", "Discounter Plus"],
                     ["POS {amt} K1 {d} {t} {p} Filiale {n}", "Einkauf {p} Kartenzahlung"], (-120, -5)),
    "Essen": (["Pizzeria Napoli", "Sushi Garden", "Burger Werk", "Gasthaus Linde", "Thai Kitchen"],
              ["POS {amt} K1 {d} {t} {p}", "{p} Restaurant Kartenzahlung"], (-80, -10)),
    "Oeffis": (["Stadtwerke Verkehr", "Bahn Ticketshop", "Citybus App"],
               ["E-COMM {amt} K1 {d} {t} {p} Ticket {n}", "{p} Fahrkarte"], (-60, -2)),
    "Bargeld": (["Bankomat"], ["Auszahlung Bankomat {d} {t} Automat {n}", "Bargeldbehebung {n}"], (-300, -20)),
    "Strom": (["Energie Nord GmbH", "Stromversorger Ost"], ["Teilbetrag Strom Vertrag {n}", "Abschlag Strom {d}"], (-90, -40)),
    "Internet": (["Netzfunk Telekom", "Glasfaser Plus"], ["Rechnung Internet Kundennr {n}", "Mobilfunk Rechnung {n}"], (-60, -15)),
    "Einkommen": (["Beispiel Arbeitgeber GmbH"], ["Gehalt {m}", "Lohn Gehalt Monat {m}"], (2000, 3500)),
    "Gesundheit": (["Apotheke Zum Hirschen", "Praxis Dr Muster"], ["POS {amt} {p}", "Honorarnote {n}"], (-150, -5)),
    "Auto": (["Autohaus Beispiel", "Werkstatt Schraube"], ["Service Rechnung {n}", "Reparatur {p}"], (-600, -50)),
    "Tanken": (["Tankstelle Blitz", "Fuel Point"], ["POS {amt} K1 {d} {t} {p} Zapfsaeule {n}"], (-90, -30)),
    "Versicherung": (["Musterversicherung AG", "Sicher Leben AG"], ["Polizze {n} Praemie", "Folgepraemie Vertrag {n}"], (-120, -20)),
    "Kleidung": (["Modehaus Faden", "Sport Outlet"], ["POS {amt} {p}", "Online Bestellung {p} {n}"], (-200, -15)),
    "Freizeit": (["Kino Lichtspiele", "Hallenbad Mitte", "Kletterhalle Gipfel"], ["POS {amt} {p} Eintritt", "{p} Ticket {n}"], (-50, -5)),
    "Miete": (["Hausverwaltung Beispiel"], ["Miete {m} Top {n}", "Mietzahlung {m}"], (-1100, -800)),
}

ACCOUNTS = ["Girokonto", "Kreditkarte"]


def fake_iban():
    return "XX00" + "".join(random.choice("0123456789") for _ in range(16))


def main(out):
    rows = []
    start = date(2023, 1, 1)
    for _ in range(1500):
        cat = random.choice(list(CATEGORIES))
        partners, refs, (lo, hi) = CATEGORIES[cat]
        p = random.choice(partners)
        day = start + timedelta(days=random.randint(0, 600))
        amt = round(random.uniform(lo, hi), 2)
        ref = random.choice(refs).format(
            amt=f"{abs(amt):.2f}".replace(".", ","), d=day.strftime("%d.%m."),
            t=f"{random.randint(6, 22):02d}:{random.randint(0, 59):02d}",
            p=p.upper(), n=random.randint(1000, 99999), m=day.strftime("%m/%Y"))
        rows.append({
            "booking": day.isoformat(), "partnerName": p, "partnerAccount.iban": fake_iban(),
            "amount.value": amt, "amount.currency": "EUR", "reference": ref,
            "account": random.choice(ACCOUNTS), "category": cat,
        })
    df = pd.DataFrame(rows).sort_values("booking")
    df.to_excel(out, sheet_name="Sheet1", index=False)
    print(f"wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
