# AXON Provider ABI v0.1 — vCPNPU / vCXL

This repository exposes a process-level Provider ABI shim:

    python provider_abi.py <action> --request-json '<json>'

Supported actions:

- describe
- memory_prepare
- memory_release
- execute
- telemetry

## Evidence boundary

The provider is an executable process interface, but its computation is still
the deterministic simulation already implemented in:

    simulate_performance_rig.py

Therefore every response remains:

    simulated

This must not be described as:

- physical CXL execution;
- physical CPNPU execution;
- a measured Windows daemon;
- measured GPU DMA/prefetch behavior.

## vCXL actions

`memory_prepare` and `memory_release` maintain simulated lease metadata in a
state JSON file.

They do not move physical pages or program a CXL controller.

## vCPNPU execute

`execute` calls the existing `simulate_axon_engine()` model.

The default profile is a Qwen-32B-like virtual workload on a 12GB/32GB,
PCIe-4-like synthetic configuration inherited from the repository's existing
simulation assumptions.

Callers may provide an explicit `simulation_profile` in the Provider ABI
payload to change those assumptions.

## Why this exists

AXON Assembly can test its real external-process transport and Provider ABI
against the same interface that future vCPNPU/vCXL engine binaries can adopt.

When an actual engine is source-audited, only the provider implementation and
evidence class need to change; Assembly's execution contract does not.

## Phase 36 execution identity and durable replay

The simulated vCXL/vCPNPU Provider requires both `execution_key` and
`idempotency_key` for the protected side-effect actions:

- `memory_prepare`
- `memory_release`
- `execute`

Both identities are canonical lowercase SHA-256 text.

The Provider advertises the same Phase-36 contract consumed by AXON Assembly:

```json
{
  "idempotency": {
    "mode": "durable-result-cache",
    "actions": ["execute", "memory_prepare", "memory_release"],
    "replay_response": true,
    "ambiguous_inflight_blocks_retry": true
  }
}
```

The SQLite durable journal follows:

    NEW -> INFLIGHT -> Provider operation -> COMPLETED + full response

The INFLIGHT claim is committed before the Provider operation. A completed
duplicate with the same execution/action/payload binding returns the stored
response with `replayed=true` and is not intentionally re-executed.

If the Provider disappears after the durable INFLIGHT claim but before
completion evidence is recorded, a duplicate remains ambiguous and automatic
retry is blocked.

Reuse of one `idempotency_key` with a different execution key, action, or
canonical request payload is rejected as a conflict.

This is a control-safety property only. The replay journal is not claimed to be
transactionally atomic with a future physical CXL, DMA, GPU, or CPNPU side
effect. The current provider remains a SIMULATED vCXL/vCPNPU path and does not
prove physical hardware execution or promote SIMULATED evidence to MEASURED.
