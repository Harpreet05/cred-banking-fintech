from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import loan_applications


ESCALATION_THRESHOLD = 0.90
MAX_RECENCY_DAYS = 30


def check_loan_application_status(record_id: str) -> dict:
    """Return loan status, amount, and a fraud/recency escalation score."""
    application = next(
        (record for record in loan_applications if record["record_id"] == record_id),
        None,
    )
    if application is None:
        raise ValueError(f"Loan application not found: {record_id}")

    fraud_signal = float(application["flagged_for_fraud_review"])
    recency_signal = min(application["days_since_created"], MAX_RECENCY_DAYS) / MAX_RECENCY_DAYS
    escalation_score = 0.6 * fraud_signal + 0.4 * recency_signal

    return {
        "record_id": record_id,
        "status": application["status"],
        "loan_amount_inr": application["loan_amount_inr"],
        "escalation_score": round(escalation_score, 4),
        "escalate": escalation_score > ESCALATION_THRESHOLD,
    }


if __name__ == "__main__":
    record_id = loan_applications[4]["record_id"]
    print(check_loan_application_status(record_id))
