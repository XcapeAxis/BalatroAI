# Architecture and Data Flow (P25)

## System Overview

```mermaid
flowchart LR
  subgraph Real[Real Runtime]
    G[Balatro Game]
    R[balatrobot RPC]
    G --> R
  end

  subgraph Canonical[Oracle and Canonicalization]
    O[Oracle Collector]
    C[canonical state_v1/action_v1/trace_v1]
    D[oracle-sim diff]
    O --> C --> D
  end

  subgraph Sim[Simulator]
    S[sim engine]
  end

  subgraph Train[Trainer Pipelines]
    RO[rollout]
    TB[train_bc / train_pv]
    TR[train_rl]
    EV[eval]
    IN[infer_assistant]
    RO --> TB
    TB --> TR
    TR --> EV --> IN
  end

  subgraph Ops[Experiment Ops]
    ORCH[P22+ orchestrator]
    CAMP[P24 campaign manager]
    TRI[triage/bisect]
    RANK[ranking]
    CC[champion/candidate]
    ART[artifacts + reports]
    ORCH --> CAMP --> TRI --> RANK --> CC
    CAMP --> ART
  end

  R --> O
  C --> S
  S --> RO
  EV --> ORCH
```

## Data Flow

1. Real or simulated state transitions are collected into canonical traces.
2. Canonical traces and simulator traces are diffed for parity confidence.
3. Rollout generates datasets for BC/PV/RL training.
4. Evaluation writes summary metrics and run manifests.
5. Orchestrator/campaign manager executes matrix + seed policies.
6. Telemetry, summaries, and gate reports are persisted to `docs/artifacts/*`.
7. Triage/flake/ranking outputs feed champion-candidate decisions.
