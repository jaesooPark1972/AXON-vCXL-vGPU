import tempfile
import unittest
from pathlib import Path

from provider_dedup import DurableReplayStore, ProviderReplayConflict


def protected_request():
    return {
        "schema": "axon.provider-request/v0.1",
        "request_id": "r1",
        "action": "execute",
        "payload": {"workload": "w"},
        "execution_key": "a" * 64,
        "idempotency_key": "b" * 64,
    }


class ProviderDedupTests(unittest.TestCase):
    def test_inflight_claim_survives_operation_crash_and_blocks_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DurableReplayStore(Path(tmp) / "dedup.sqlite3")
            calls = []

            def crash():
                calls.append("crash")
                raise KeyboardInterrupt("simulated process death")

            with self.assertRaises(KeyboardInterrupt):
                store.run(protected_request(), crash)

            def must_not_run():
                calls.append("rerun")
                return {"ok": True}

            with self.assertRaisesRegex(ProviderReplayConflict, "INFLIGHT"):
                store.run(protected_request(), must_not_run)

            self.assertEqual(calls, ["crash"])

    def test_completed_response_is_replayed_without_reexecution(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DurableReplayStore(Path(tmp) / "dedup.sqlite3")
            calls = []

            def execute():
                calls.append("run")
                return {
                    "schema": "axon.provider-response/v0.1",
                    "request_id": "r1",
                    "provider_id": "p",
                    "action": "execute",
                    "ok": True,
                    "evidence": {"class": "simulated"},
                    "result": {"value": 1},
                    "telemetry": {},
                    "error": None,
                }

            first = store.run(protected_request(), execute)
            second = store.run(protected_request(), execute)

            self.assertFalse(first["replayed"])
            self.assertTrue(second["replayed"])
            self.assertEqual(calls, ["run"])


if __name__ == "__main__":
    unittest.main()
