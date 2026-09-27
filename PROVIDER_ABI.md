# AXON Provider ABI v0.1 — vCPNPU / vCXL

This repository now exposes a process-level Provider ABI shim:

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

AXON Assembly can now test its real external-process transport and Provider ABI
against the same interface that future vCPNPU/vCXL engine binaries can adopt.

When an actual engine is source-audited, only the provider implementation and
evidence class need to change; Assembly's execution contract does not.

## Phase 36 execution identity and durable replay

The simulated vCXL/vCPNPU provider now requires an ASM execution identity for `memory_prepare`, `memory_release`, and `execute`.

It advertises durable replay with `side_effect_atomic=false`. The replay cache is SQLite-backed and uses `BEGIN IMMEDIATE` so competing Provider processes with the same operation key serialize. The first response is stored; later identical calls return it with `replayed=true`.

This is a control-safety property only. The provider remains a SIMULATED vCXL/vCPNPU path and does not prove physical CXL, DMA, GPU, or CPNPU execution.

