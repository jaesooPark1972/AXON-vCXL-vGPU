from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from provider_dedup import DurableReplayStore, replay_db_for_state
from simulate_performance_rig import HardwareSpec, ModelSpec, simulate_axon_engine


PROVIDER_ABI_SCHEMA = "axon.provider-abi/v0.1"
PROVIDER_REQUEST_SCHEMA = "axon.provider-request/v0.1"
PROVIDER_RESPONSE_SCHEMA = "axon.provider-response/v0.1"
PROVIDER_ABI_VERSION = "0.1"
PROVIDER_ID = "axon-vcxl-vcpnpu-provider"

IDEMPOTENT_ACTIONS = {
    "memory_prepare",
    "memory_release",
    "execute",
}

ACTIONS = (
    "describe",
    "memory_prepare",
    "memory_release",
    "execute",
    "telemetry",
)


def provider_manifest() -> dict[str, object]:
    return {
        "schema": PROVIDER_ABI_SCHEMA,
        "abi_version": PROVIDER_ABI_VERSION,
        "provider_id": PROVIDER_ID,
        "provider_kind": "virtual",
        "subsystem": "axon-vcxl-vcpnpu-simulation",
        "actions": list(ACTIONS),
        "state_mode": "stateful",
        "transport": {
            "kind": "process-json-argv",
            "request_encoding": "json",
            "response_encoding": "json",
        },
        "idempotency": {
            "mode": "durable-result-cache",
            "actions": sorted(IDEMPOTENT_ACTIONS),
            "replay_response": True,
            "ambiguous_inflight_blocks_retry": True,
        },
        "evidence": {
            "class": "simulated",
            "note": (
                "Provider executes the repository's deterministic vCPNPU/vCXL "
                "simulation model; it is not a physical or live Windows engine."
            ),
        },
        "limitations": [
            "memory_prepare/memory_release persist only virtual lease metadata.",
            "execute calls simulate_axon_engine() from simulate_performance_rig.py.",
            "No physical CXL link, DMA engine, vGPU daemon, or CPNPU ASIC is controlled.",
            (
                "SQLite durable replay serializes Provider ABI operations and returns "
                "recorded responses; future physical-device side effects require their "
                "own transaction/idempotency integration."
            ),
        ],
    }


def handle_request(
    request: Mapping[str, Any],
    *,
    state_path: str | Path,
) -> dict[str, object]:
    request_id = _non_empty_string(request.get("request_id"), "request_id")
    action = _non_empty_string(request.get("action"), "action")
    try:
        _validate_request(request)
    except Exception as exc:
        return _response(
            request_id=request_id,
            action=action,
            ok=False,
            result={},
            telemetry={},
            execution_key=None,
            idempotency_key=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    def operation() -> dict[str, object]:
        try:
            return _execute_validated_request(
                request,
                state_path=Path(state_path),
            )
        except Exception as exc:
            return _response(
                request_id=request_id,
                action=action,
                ok=False,
                result={},
                telemetry={},
                execution_key=_optional_string(request.get("execution_key")),
                idempotency_key=_optional_string(request.get("idempotency_key")),
                error=f"{type(exc).__name__}: {exc}",
            )

    if (
        action in IDEMPOTENT_ACTIONS
        and request.get("execution_key") is not None
        and request.get("idempotency_key") is not None
    ):
        try:
            return DurableReplayStore(
                replay_db_for_state(state_path)
            ).run(request, operation)
        except Exception as exc:
            return _response(
                request_id=request_id,
                action=action,
                ok=False,
                result={},
                telemetry={},
                execution_key=_optional_string(request.get("execution_key")),
                idempotency_key=_optional_string(request.get("idempotency_key")),
                error=f"{type(exc).__name__}: {exc}",
            )
    return operation()


def _execute_validated_request(
    request: Mapping[str, Any],
    *,
    state_path: Path,
) -> dict[str, object]:
    request_id = _non_empty_string(request.get("request_id"), "request_id")
    action = _non_empty_string(request.get("action"), "action")
    payload = request.get("payload", {})
    assert isinstance(payload, dict)

    if action == "describe":
        result = provider_manifest()
        telemetry: dict[str, object] = {}
    elif action == "memory_prepare":
        result = _memory_prepare(payload, request, state_path)
        telemetry = {}
    elif action == "memory_release":
        result = _memory_release(payload, request, state_path)
        telemetry = {}
    elif action == "execute":
        result, telemetry = _execute(payload, state_path)
    elif action == "telemetry":
        state = _load_state(state_path)
        result = {
            "active_memory_leases": dict(state.get("memory_leases", {})),
            "state_path": str(state_path),
        }
        telemetry = dict(state.get("last_telemetry", {}))
    else:
        raise ValueError(f"unsupported action: {action}")

    return _response(
        request_id=request_id,
        action=action,
        ok=True,
        result=result,
        telemetry=telemetry,
        execution_key=_optional_string(request.get("execution_key")),
        idempotency_key=_optional_string(request.get("idempotency_key")),
    )

def _memory_prepare(
    payload: Mapping[str, Any],
    request: Mapping[str, Any],
    state_path: Path,
) -> dict[str, object]:
    workload = _non_empty_string(payload.get("workload"), "workload")
    memory_node = _non_empty_string(payload.get("memory_node"), "memory_node")
    memory_kind = _non_empty_string(payload.get("memory_kind"), "memory_kind")
    required_bytes = int(payload.get("required_memory_bytes", 0) or 0)
    if required_bytes < 0:
        raise ValueError("required_memory_bytes must be non-negative")

    key = _operation_key("memory_prepare", request)
    state = _load_state(state_path)
    operations = state.setdefault("operations", {})
    leases = state.setdefault("memory_leases", {})
    assert isinstance(operations, dict)
    assert isinstance(leases, dict)

    if key in operations:
        saved = operations[key]
        if not isinstance(saved, dict):
            raise ValueError("corrupt provider state")
        return dict(saved)

    lease_id = f"{workload}:{memory_node}"
    leases[lease_id] = {
        "workload": workload,
        "memory_node": memory_node,
        "memory_kind": memory_kind,
        "required_memory_bytes": required_bytes,
        "status": "prepared-simulated",
    }
    result = {
        "lease_id": lease_id,
        "memory_node": memory_node,
        "memory_kind": memory_kind,
        "required_memory_bytes": required_bytes,
        "status": "prepared-simulated",
    }
    operations[key] = result
    _save_state(state_path, state)
    return result


def _memory_release(
    payload: Mapping[str, Any],
    request: Mapping[str, Any],
    state_path: Path,
) -> dict[str, object]:
    workload = _non_empty_string(payload.get("workload"), "workload")
    memory_node = _non_empty_string(payload.get("memory_node"), "memory_node")
    key = _operation_key("memory_release", request)

    state = _load_state(state_path)
    operations = state.setdefault("operations", {})
    leases = state.setdefault("memory_leases", {})
    assert isinstance(operations, dict)
    assert isinstance(leases, dict)

    if key in operations:
        saved = operations[key]
        if not isinstance(saved, dict):
            raise ValueError("corrupt provider state")
        return dict(saved)

    lease_id = f"{workload}:{memory_node}"
    existed = leases.pop(lease_id, None) is not None
    result = {
        "lease_id": lease_id,
        "released": existed,
        "status": "released-simulated" if existed else "not-found",
    }
    operations[key] = result
    _save_state(state_path, state)
    return result


def _execute(
    payload: Mapping[str, Any],
    state_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    workload = str(payload.get("workload") or "asm-virtual-workload")
    profile = payload.get("simulation_profile", {})
    if profile is None:
        profile = {}
    if not isinstance(profile, dict):
        raise TypeError("simulation_profile must be an object when provided")

    hw = HardwareSpec(
        name=str(profile.get("hardware_name", "AXON Provider Virtual GPU")),
        vram_bytes=int(profile.get("vram_bytes", 12 * 1024 * 1024 * 1024)),
        system_ram_bytes=int(
            profile.get("system_ram_bytes", 32 * 1024 * 1024 * 1024)
        ),
        pcie_bw_gb_s=float(profile.get("pcie_bw_gb_s", 15.75)),
        gpu_tflops_fp16=float(profile.get("gpu_tflops_fp16", 29.0)),
    )

    num_layers = int(profile.get("num_layers", 64))
    param_count_b = float(profile.get("param_count_b", 32.5))
    model_footprint_bytes = int(
        profile.get(
            "model_footprint_bytes",
            int(32.5 * 0.26 * 1024 * 1024 * 1024),
        )
    )
    if num_layers <= 0 or model_footprint_bytes <= 0:
        raise ValueError("simulation profile layers/footprint must be positive")

    model = ModelSpec(
        name=str(profile.get("model_name", workload)),
        num_layers=num_layers,
        param_count_b=param_count_b,
        bytes_per_layer=max(1, model_footprint_bytes // num_layers),
        flops_per_token_layer=float(
            profile.get("flops_per_token_layer", 1_200_000_000)
        ),
        sparsity_ratio=float(profile.get("sparsity_ratio", 0.40)),
    )

    tokens = int(profile.get("tokens_to_generate", 25))
    if tokens <= 0:
        raise ValueError("tokens_to_generate must be positive")

    simulated = simulate_axon_engine(hw, model, tokens_to_generate=tokens)
    telemetry = {
        "provider_id": PROVIDER_ID,
        "provider_kind": "virtual",
        "evidence_class": "simulated",
        "throughput_tok_s": simulated["throughput_tok_s"],
        "avg_latency_ms": simulated["avg_latency_ms"],
        "bus_stall_pct": simulated["bus_stall_pct"],
        "oom_occurred": simulated["oom_occurred"],
        "latency_hidden_pct": simulated["latency_hidden_pct"],
        "hardware_profile": hw.name,
        "model_profile": model.name,
    }

    state = _load_state(state_path)
    state["last_telemetry"] = telemetry
    _save_state(state_path, state)

    return {
        "workload": workload,
        "backend_node": payload.get("backend_node"),
        "memory_nodes": list(payload.get("memory_nodes", [])),
        "simulation": simulated,
    }, telemetry


def _validate_request(request: Mapping[str, Any]) -> None:
    if request.get("schema") != PROVIDER_REQUEST_SCHEMA:
        raise ValueError(f"schema must be {PROVIDER_REQUEST_SCHEMA!r}")
    _non_empty_string(request.get("request_id"), "request_id")
    action = _non_empty_string(request.get("action"), "action")
    if action not in ACTIONS:
        raise ValueError(f"unsupported action: {action}")
    payload = request.get("payload", {})
    if not isinstance(payload, dict):
        raise TypeError("payload must be an object")
    key = request.get("idempotency_key")
    execution_key = request.get("execution_key")
    if (key is None) != (execution_key is None):
        raise ValueError(
            "execution_key and idempotency_key must be supplied together"
        )
    if execution_key is not None:
        _sha256(execution_key, "execution_key")
        _sha256(key, "idempotency_key")


def _response(
    *,
    request_id: str,
    action: str,
    ok: bool,
    result: Mapping[str, object],
    telemetry: Mapping[str, object],
    execution_key: str | None = None,
    idempotency_key: str | None = None,
    replayed: bool = False,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "schema": PROVIDER_RESPONSE_SCHEMA,
        "request_id": request_id,
        "provider_id": PROVIDER_ID,
        "action": action,
        "ok": ok,
        "evidence": {
            "class": "simulated",
            "note": (
                "Response is produced by the repository's deterministic "
                "vCPNPU/vCXL simulation provider."
            ),
        },
        "result": dict(result),
        "telemetry": dict(telemetry),
        "execution_key": execution_key,
        "idempotency_key": idempotency_key,
        "replayed": replayed,
        "error": error,
    }


def _operation_key(action: str, request: Mapping[str, Any]) -> str:
    key = request.get("idempotency_key") or request.get("request_id")
    return f"{action}:{key}"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "memory_leases": {},
            "operations": {},
            "last_telemetry": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("provider state root must be an object")
    return data


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _sha256(value: object, field_name: str) -> str:
    text = _non_empty_string(value, field_name).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return text


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, "optional_string")


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field_name} must be a non-empty string")
    return value.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python provider_abi.py",
        description="AXON vCPNPU/vCXL Provider ABI v0.1 simulated process provider",
    )
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--request-json", required=True)
    parser.add_argument(
        "--state",
        default=".axon_vcxl_vcpnpu_provider_state.json",
        help="persistent provider state JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        request = json.loads(args.request_json)
        if not isinstance(request, dict):
            raise TypeError("request JSON root must be an object")
        if request.get("action") != args.action:
            raise ValueError("CLI action does not match request action")
        response = handle_request(request, state_path=args.state)
    except Exception as exc:
        response = _response(
            request_id="invalid-request",
            action=args.action,
            ok=False,
            result={},
            telemetry={},
            error=f"{type(exc).__name__}: {exc}",
        )
    print(json.dumps(response, sort_keys=True))
    return 0 if response["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
