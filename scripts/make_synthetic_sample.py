#!/usr/bin/env python3
"""Create a deterministic, synthetic loan CSV with the audited 24-column schema."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date
from pathlib import Path


SCHEMA = [
    "",
    "id",
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "loan_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "mths_since_last_delinq",
    "open_acc",
    "revol_bal",
    "total_acc",
    "out_prncp",
    "total_pymnt",
    "total_rec_prncp",
    "total_rec_int",
]


def _installment(loan_amount: float, term_months: int, rate: float) -> float:
    monthly_rate = rate / 12.0
    if monthly_rate == 0:
        return loan_amount / term_months
    factor = (1 + monthly_rate) ** term_months
    return loan_amount * monthly_rate * factor / (factor - 1)


def generate_rows(n_rows: int = 200, seed: int = 42) -> list[dict[str, object]]:
    rng = random.Random(seed)
    statuses = (
        ["Fully Paid"] * 90
        + ["Charged Off"] * 45
        + ["Default"] * 5
        + ["In Grace Period"] * 7
        + ["Late (31-120 days)"] * 8
        + ["Late (16-30 days)"] * 5
        + ["Current"] * 30
        + [""] * 10
    )
    if n_rows != len(statuses):
        statuses = (statuses * ((n_rows // len(statuses)) + 1))[:n_rows]
    rng.shuffle(statuses)

    terms = ["36 months", "60 months"]
    homes = ["MORTGAGE", "RENT", "OWN", "OTHER"]
    purposes = [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "small_business",
        "major_purchase",
        "car",
        "medical",
        "other",
    ]
    states = ["CA", "NY", "TX", "FL", "IL", "PA", "WA", "CO", "GA", "MA"]
    rows: list[dict[str, object]] = []
    for row_number in range(n_rows):
        amount = float(rng.randrange(1000, 35001, 500))
        term = rng.choice(terms)
        term_months = int(term.split()[0])
        rate = rng.uniform(0.06, 0.28)
        income = float(rng.randrange(24000, 150001, 1000))
        installment = _installment(amount, term_months, rate)
        has_delinq = rng.random() < 0.40
        earliest_year = rng.randint(1985, 2008)
        earliest_month = rng.randint(1, 12)
        earliest_day = rng.randint(1, 28)
        row = {
            "": row_number,
            "id": f"synthetic-{row_number:05d}",
            "loan_amnt": f"{amount:.0f}",
            "funded_amnt": f"{amount:.0f}",
            "term": term,
            "int_rate": f"{rate * 100:.2f}%",
            "installment": f"{installment:.2f}",
            "emp_length": rng.choice(
                ["< 1 year", "1 year", "2 years", "5 years", "10+ years", "n/a"]
            ),
            "home_ownership": rng.choice(homes),
            "annual_inc": f"{income:.0f}",
            "loan_status": statuses[row_number],
            "purpose": rng.choice(purposes),
            "addr_state": rng.choice(states),
            "dti": f"{rng.uniform(0, 34.9):.2f}",
            "delinq_2yrs": rng.randint(0, 3),
            "earliest_cr_line": date(earliest_year, earliest_month, earliest_day).isoformat()
            + " 12:00",
            "mths_since_last_delinq": str(rng.randint(1, 122)) if has_delinq else "",
            "open_acc": rng.randint(2, 25),
            "revol_bal": rng.randint(0, 90000),
            "total_acc": rng.randint(5, 55),
            "out_prncp": f"{rng.uniform(0, amount):.2f}",
            "total_pymnt": f"{rng.uniform(0, amount * 1.3):.2f}",
            "total_rec_prncp": f"{rng.uniform(0, amount):.2f}",
            "total_rec_int": f"{rng.uniform(0, amount * 0.3):.2f}",
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(generate_rows(n_rows=args.rows, seed=args.seed))
    print(f"Wrote {args.rows} synthetic rows to {args.output}")


if __name__ == "__main__":
    main()
