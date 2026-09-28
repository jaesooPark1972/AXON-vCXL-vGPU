import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path

from provider_abi import (
    IDEMPOTENT_ACTIONS,
    PROVIDER_ID,
    PROVIDER_REQUEST_SCHEMA,
    handle_request,
    main,
    provider_manifest,
)


def request(request_id, action, payload=None, key=None, execution_key=None):
    def digest(value, default):
        raw = default if value is None else value
        if isinstance(raw, str) and len(raw) == 64 and all(
            ch in "0123456789abcdef" for ch in raw
        ):
            return raw
        return hashlib.sha256(str(raw).encode("utf-8")).hexdigest()

    data = {
        "schema": PROVIDER_REQUEST_SCHEMA,
        "request_id": request_id,
        "action": action,
        "payload": payload or {},
    }
    if action in IDEMPOTENT_ACTIONS:
        data["execution_key"] = digest(execution_key, "b" * 64)
        data["idempotency_key"] = digest(key, f"{action}-default")
    else:
        if execution_key is not None:
            data["execution_key"] = execution_key
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
        self.assertEqual(
            manifest["idempotency"]["mode"],
            "durable-result-cache",
        )
        self.assertTrue(manifest["idempotency"]["replay_response"])
        self.assertTrue(
            manifest["idempotency"]["ambiguous_inflight_blocks_retry"]
        )

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
                key="lease-1-release",
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

    def test_execute_duplicate_replays_recorded_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            req = request(
                "e-replay",
                "execute",
                {
                    "workload": "qwen-like",
                    "backend_node": "hardware:electronic-cpnpu",
                    "memory_nodes": ["memory:shared-cxl"],
                },
                key="execute-replay-key",
            )
            first = handle_request(req, state_path=state)
            second = handle_request(req, state_path=state)
            self.assertTrue(first["ok"])
            self.assertFalse(first["replayed"])
            self.assertTrue(second["ok"])
            self.assertTrue(second["replayed"])
            self.assertEqual(second["result"], first["result"])
            self.assertEqual(second["execution_key"], "b" * 64)

    def test_idempotency_key_reuse_for_different_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            first = request(
                "m1",
                "memory_prepare",
                {
                    "workload": "w1",
                    "memory_node": "memory:shared-cxl",
                    "memory_kind": "cxl",
                    "required_memory_bytes": 1024,
                },
                key="same-key",
            )
            second = request(
                "m2",
                "memory_prepare",
                {
                    "workload": "w2",
                    "memory_node": "memory:shared-cxl",
                    "memory_kind": "cxl",
                    "required_memory_bytes": 2048,
                },
                key="same-key",
            )
            self.assertTrue(handle_request(first, state_path=state)["ok"])
            rejected = handle_request(second, state_path=state)
            self.assertFalse(rejected["ok"])
            self.assertIn("different provider operation", rejected["error"])

    def test_concurrent_memory_prepare_executes_once_and_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            req = request(
                "m-concurrent",
                "memory_prepare",
                {
                    "workload": "w",
                    "memory_node": "memory:shared-cxl",
                    "memory_kind": "cxl",
                    "required_memory_bytes": 1024,
                },
                key="concurrent-memory-key",
            )
            barrier = threading.Barrier(3)
            results = []
            lock = threading.Lock()

            def worker():
                barrier.wait()
                response = handle_request(req, state_path=state)
                with lock:
                    results.append(response)

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join()

            self.assertEqual(len(results), 2)
            first_exec = [
                item for item in results
                if item["ok"] and not item["replayed"]
            ]
            self.assertEqual(len(first_exec), 1)
            duplicate = next(
                item for item in results if item is not first_exec[0]
            )
            if duplicate["ok"]:
                self.assertTrue(duplicate["replayed"])
                self.assertEqual(
                    duplicate["result"],
                    first_exec[0]["result"],
                )
            else:
                self.assertIn("INFLIGHT", duplicate["error"])

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


    def test_protected_execute_is_durably_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            protected = request(
                "protected-1",
                "execute",
                {
                    "workload": "protected-virtual",
                    "backend_node": "hardware:electronic-cpnpu",
                    "memory_nodes": ["memory:shared-cxl"],
                },
                key="b" * 64,
            )
            protected["execution_key"] = "a" * 64

            first = handle_request(protected, state_path=state)
            second = handle_request(protected, state_path=state)

            self.assertTrue(first["ok"])
            self.assertTrue(second["ok"])
            self.assertFalse(first["replayed"])
            self.assertTrue(second["replayed"])
            self.assertEqual(first["execution_key"], "a" * 64)
            self.assertEqual(second["idempotency_key"], "b" * 64)
            self.assertEqual(first["result"], second["result"])

    def test_protected_key_payload_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            first = request(
                "protected-1",
                "memory_prepare",
                {
                    "workload": "w1",
                    "memory_node": "memory:shared-cxl",
                    "memory_kind": "cxl",
                    "required_memory_bytes": 1024,
                },
                key="d" * 64,
            )
            first["execution_key"] = "c" * 64
            changed = dict(first)
            changed["request_id"] = "protected-2"
            changed["payload"] = dict(first["payload"], workload="w2")

            self.assertTrue(handle_request(first, state_path=state)["ok"])
            conflict = handle_request(changed, state_path=state)
            self.assertFalse(conflict["ok"])
            self.assertIn("different provider operation", conflict["error"])

    def test_manifest_advertises_asm_idempotency_contract(self):
        contract = provider_manifest()["idempotency"]
        self.assertEqual(contract["mode"], "durable-result-cache")
        self.assertIn("execute", contract["actions"])
        self.assertTrue(contract["replay_response"])
        self.assertTrue(contract["ambiguous_inflight_blocks_retry"])


if __name__ == "__main__":
    unittest.main()
