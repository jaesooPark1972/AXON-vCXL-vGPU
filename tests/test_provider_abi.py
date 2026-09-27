import json
import tempfile
import unittest
from pathlib import Path

from provider_abi import (
    PROVIDER_ID,
    PROVIDER_REQUEST_SCHEMA,
    handle_request,
    main,
    provider_manifest,
)


def request(request_id, action, payload=None, key=None):
    data = {
        "schema": PROVIDER_REQUEST_SCHEMA,
        "request_id": request_id,
        "action": action,
        "payload": payload or {},
    }
    if key is not None:
        data["idempotency_key"] = key
    return data


class ProviderABITests(unittest.TestCase):
    def test_manifest_is_virtual_and_simulated(self):
        manifest = provider_manifest()
        self.assertEqual(manifest["provider_id"], PROVIDER_ID)
        self.assertEqual(manifest["provider_kind"], "virtual")
        self.assertEqual(manifest["evidence"]["class"], "simulated")
        self.assertIn("memory_prepare", manifest["actions"])
        self.assertIn("execute", manifest["actions"])

    def test_memory_prepare_release_is_persistent_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            prepare = request(
                "m1",
                "memory_prepare",
                {
                    "workload": "w",
                    "memory_node": "memory:shared-cxl",
                    "memory_kind": "cxl",
                    "required_memory_bytes": 1024,
                },
                key="lease-1",
            )
            first = handle_request(prepare, state_path=state)
            second = handle_request(prepare, state_path=state)
            self.assertTrue(first["ok"])
            self.assertTrue(second["ok"])
            self.assertEqual(first["result"]["lease_id"], second["result"]["lease_id"])

            release = request(
                "m2",
                "memory_release",
                {
                    "workload": "w",
                    "memory_node": "memory:shared-cxl",
                    "memory_kind": "cxl",
                },
                key="lease-1",
            )
            released = handle_request(release, state_path=state)
            self.assertTrue(released["ok"])
            self.assertTrue(released["result"]["released"])

    def test_execute_calls_existing_simulation_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            response = handle_request(
                request(
                    "e1",
                    "execute",
                    {
                        "workload": "qwen-like",
                        "backend_node": "hardware:electronic-cpnpu",
                        "memory_nodes": ["memory:shared-cxl"],
                    },
                ),
                state_path=state,
            )
            self.assertTrue(response["ok"])
            self.assertEqual(response["evidence"]["class"], "simulated")
            self.assertGreater(response["telemetry"]["throughput_tok_s"], 0)
            self.assertGreater(response["telemetry"]["avg_latency_ms"], 0)

            telemetry = handle_request(
                request("t1", "telemetry", {}),
                state_path=state,
            )
            self.assertEqual(
                telemetry["telemetry"]["provider_id"],
                PROVIDER_ID,
            )

    def test_bad_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = request("bad", "describe")
            raw["schema"] = "wrong"
            response = handle_request(
                raw,
                state_path=Path(tmp) / "state.json",
            )
            self.assertFalse(response["ok"])
            self.assertIn("schema", response["error"])


if __name__ == "__main__":
    unittest.main()
