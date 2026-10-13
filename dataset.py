from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List

category_weights = {
    "Personal Loan" : 0.30,
    "Home Loan" : 0.22,
    "Auto Loan" : 0.18,
    "Education Loan" : 0.15,
    "Business Loan" : 0.15,
}

status_weights = {
    "Submitted" : 0.30,
    "Under Review" : 0.25,
    "Approved" : 0.20,
    "Rejected" : 0.15,
    "Disbursed" : 0.10,
}

category_amount_ranges = {
    "Personal Loan": (50_000, 20_00_000),
    "Home Loan": (4_00_000, 80_00_000),
    "Auto Loan": (1_00_000, 30_00_000),
    "Education Loan": (1_00_000, 25_00_000),
    "Business Loan": (3_00_000, 60_00_000),
}

categories = list(category_weights.keys())
statuses = list(status_weights.keys())

def weighted_choice(weights: Dict[str, float], rng: random.Random) -> str:
    values = list(weights.keys())
    probs = list(weights.values())
    return rng.choices(values, weights=probs, k=1)[0]

def generate_loan_applications(count = 50, seed = 75) -> List[Dict[str,object]]:
    rng = random.Random(seed)
    records = []

    for i in range(count):
        category = weighted_choice(category_weights, rng)
        status = weighted_choice(status_weights, rng)
        min_amt, max_amt = category_amount_ranges[category]
        loan_amount = rng.randint(min_amt, max_amt)
        days_since_created = rng.randint(0, 30)

        if category in {"Auto Loan", "Business Loan"}:
            flagged_prob = 0.20
        elif category == "Home Loan":
            flagged_prob = 0.18
        else:
            flagged_prob = 0.16

        flagged_for_fraud_review = rng.random() < flagged_prob

        records.append(
            {
                "record_id" : f"LOAN APPLICATION- {i+1:04d}",
                "category" : category,
                "status" : status,
                "loan_amount_inr" : loan_amount,
                "days_since_created" : days_since_created,
                "flagged_for_fraud_review" : flagged_for_fraud_review,
            }
        )

    flagged_percentage = (sum(1 for r in records if r["flagged_for_fraud_review"])/ len(records))*100
    if not 10 <= flagged_percentage <= 30:
        raise ValueError(f"Generated fraud-review share is not in the required 10–30% band: {flagged_percentage:.2f}%")
    return records

loan_applications = generate_loan_applications()


def summarize_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    category_counts = Counter(r["category"] for r in records)
    status_counts = Counter(r["status"] for r in records)
    flagged_pct = (sum(1 for r in records if r["flagged_for_fraud_review"]) / len(records)) * 100

    print("Category counts:")
    for name in categories:
        print(f"  {name}: {category_counts.get(name, 0)}")
    print("\nStatus counts:")
    for name in statuses:
        print(f"  {name}: {status_counts.get(name, 0)}")
    print(f"\nFlagged for fraud review: {flagged_pct:.2f}%")

    return {
        "category_counts": dict(category_counts),
        "status_counts": dict(status_counts),
        "flagged_pct": flagged_pct,
    }


if __name__ == "__main__":
    summarize_records(loan_applications)