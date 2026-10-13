import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import deploy
from api.structured_logging import mask_for_logging, write_request_log


class StructuredLoggingTests(unittest.TestCase):
    def test_mask_for_logging_masks_fixed_format_pii(self):
        raw = "Aadhaar 2345 6789 0123, PAN ABCDE1234F, phone 9876543210, bank account number 1234567890123456"
        masked = mask_for_logging(raw)

        self.assertIn("[AADHAAR_REDACTED]", masked)
        self.assertIn("[PAN_REDACTED]", masked)
        self.assertIn("[PHONE_REDACTED]", masked)
        self.assertIn("[ACCOUNT_REDACTED]", masked)
        self.assertNotIn("2345 6789 0123", masked)
        self.assertNotIn("ABCDE1234F", masked)
        self.assertNotIn("9876543210", masked)
        self.assertNotIn("1234567890123456", masked)

    def test_write_request_log_writes_jsonl_with_trace_and_timing(self):
        log_path = Path(self._testMethodName + ".jsonl")
        if log_path.exists():
            log_path.unlink()

        entry = write_request_log(
            path=log_path,
            method="POST",
            route="/ask",
            request_text="Aadhaar 2345 6789 0123, PAN ABCDE1234F, bank account number 1234567890123456",
            trace_id="trace-123",
            started_at=1000.0,
            finished_at=1002.5,
        )

        self.assertEqual(entry["trace_id"], "trace-123")
        self.assertEqual(entry["method"], "POST")
        self.assertEqual(entry["route"], "/ask")
        self.assertEqual(entry["duration_ms"], 2500.0)
        self.assertIn("[AADHAAR_REDACTED]", entry["request_text"])
        self.assertIn("[PAN_REDACTED]", entry["request_text"])
        self.assertIn("[ACCOUNT_REDACTED]", entry["request_text"])

        with open(log_path, "r", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["trace_id"], "trace-123")
        self.assertNotIn("2345 6789 0123", lines[0]["request_text"])
        self.assertNotIn("ABCDE1234F", lines[0]["request_text"])
        self.assertNotIn("1234567890123456", lines[0]["request_text"])

    def test_middleware_logs_unhandled_endpoint_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "requests.jsonl"

            def write_to_temp(**kwargs):
                return write_request_log(path=log_path, **kwargs)

            with patch.object(deploy, "write_request_log", side_effect=write_to_temp), patch.object(
                deploy,
                "run_conversation",
                side_effect=RuntimeError("simulated endpoint failure"),
            ):
                response = TestClient(deploy.app, raise_server_exceptions=False).post(
                    "/ask",
                    json={"query": "status?"},
                    headers={"X-Trace-Id": "failure-trace"},
                )

            entries = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(response.status_code, 500)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["trace_id"], "failure-trace")
        self.assertEqual(entries[0]["status_code"], 500)


if __name__ == "__main__":
    unittest.main()
