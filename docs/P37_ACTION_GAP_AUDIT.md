# P37 Action Trace Gap Audit

## Scope and Inputs
- trace contract/schema:
  - `sim/spec/TRACE_CONTRACT.md`
  - `sim/spec/action_v1.json`
  - `sim/spec/trace_v1.json`
- real record/replay chain:
  - `trainer/record_real_session.py`
  - `trainer/real_executor.py`
  - `trainer/real_trace_to_fixture.py`
  - `sim/tests/run_real_action_replay_fixture.py`
  - `sim/tests/run_real_drift_fixture.py`

## Gap Matrix
| action_name | real_capture | sim_apply | affected_mechanics | priority | fixture_plan |
|---|---|---|---|---|---|
| `MOVE_HAND_CARD(src_index,dst_index,index_base)` | Partial. Real runtime currently captures explicit reorder only when action is scripted (`action_sent`) or inferred from before/after snapshot in `real_trace_to_fixture.py`; no direct drag event stream. | Missing as first-class type. Current engine supports `REORDER_HAND`/`SWAP_HAND_CARDS` but not atomic move semantics. | 手里首张/顺序敏感结算，含 first-face/first-played 类机制；错误顺序会造成分数偏差。 | P0 | Add sim-only fixture with deterministic uid order assertions; add oracle-based fixture where final hand order is converted to minimal `MOVE_HAND_CARD` sequence and replay-diffed by order-sensitive scope. |
| `MOVE_JOKER(src_index,dst_index,index_base)` | Partial. Same capture pattern as hand reorder; no direct drag event source, depends on scripted action or inference. | Missing as first-class type. Engine has `REORDER_JOKERS`/`SWAP_JOKERS` only. | Joker 左右邻位语义（例如复制右侧/吞噬右侧）会直接改变分数与状态。 | P0 | Add deterministic multi-joker fixture with reorder then play; oracle fixture validates joker uid/key order parity. |
| `SWAP_HAND_CARD(i,j,index_base)` | Capturable (explicit scripted actions and inferred two-point permutation swap). | Exists as `SWAP_HAND_CARDS`; naming/contract not unified with move family. | 与 `MOVE_HAND_CARD` 等价子集；用于塔罗单步“互换”语义更直观。 | P1 | Keep compatibility fixture that replays historical `SWAP_HAND_CARDS`; add alias action test for new naming. |
| `SWAP_JOKER(i,j,index_base)` | Capturable (explicit/inferred). | Exists as `SWAP_JOKERS`; naming/contract not unified with move family. | Joker 邻位依赖机制。 | P1 | Same as above; keep legacy fixture and add alias test. |
| `CONSUMABLE_USE(kind,key,params={cards,hand_indices,joker_index,shop_index,index_base,target_side})` | Partial. Execute mode captures params only if action is scripted. Shadow/manual mode lacks target-intent capture and can only infer coarse state delta. | Weak. Engine currently treats `USE` as no-op unless `expected_context` is injected externally. | 位置依赖塔罗/幻灵（如“左边变右边”）与指定目标卡修改；target 丢失会导致不可回放。 | P0 | Build directed fixtures with explicit `params.index_base` and targets; at least one tarot-style position-dependent case and one generic consumable target case. |
| `SHOP_REROLL` | Capturable (`REROLL`). | Partial. Engine只扣钱，不原生刷新商店；依赖 oracle `rng_replay`/`expected_context` 注入。 | 商店生态与概率分布、经济路径。 | P0 | Build fixture pair: with and without replay outcomes; ensure replay path deterministic and hash-aligned. |
| `SHOP_BUY(params={shop_index/voucher_index/pack_index/index_base})` | Capturable when scripted. | Partial. `BUY` only changes state for specific injected context, no native inventory/economy complete transition. | 经济与库存、后续 pack/open 分支。 | P1 | Directed fixture asserts observable fields (money/shop slots/phase) via replay outcome context; record known native-sim gap. |
| `PACK_OPEN(params={pack_index,choice_index,index_base,skip})` | Capturable when scripted (`PACK`). | Partial. `PACK` branch exists but detailed pack-choice semantics mostly externalized to oracle context. | 补充包内概率与选择路径。 | P1 | Add fixture covering open + choose + skip, with rng outcome tokens and hash scope checks. |
| `PLAY(indices,index_base,selection_order)` | Capturable and already in schema; order currently implied by index array order. | Exists and functional. | 出牌顺序影响 first-trigger/重复触发等机制。 | P0 (re-verify) | Add regression that uses same set in different order and asserts different parity hashes where expected. |
| `DISCARD(indices,index_base,selection_order)` | Capturable and already in schema. | Exists and functional. | 弃牌消耗与回合资源节奏。 | P1 (re-verify) | Add minimal order-focused discard fixture to ensure index_base and order invariants stay stable. |
| `SHOP/PACK probability weights source contract` | Not encoded in per-step action trace; currently inferred from sampled outcomes and mechanics CSV proxy. | Partial. Sim relies on oracle replay/outcomes; native stochastic model incomplete. | shop reroll、pack 内容与稀有度分布 parity。 | P0 | Add probability audit batch: oracle vs sim replay-driven sampling with frequency tables and soft statistical warning thresholds (no hard fail yet). |

## Current Coverage Summary
- Covered today:
  - post-action trace contract is consistent (`action[N]` ↔ `state[N]`).
  - legacy position actions (`REORDER_*`, `SWAP_*`) can replay in sim and can be inferred from raw state delta in simple cases.
  - P32 hash scope already checks ordered hand/joker keys.
- Not covered or unstable:
  - no first-class `MOVE_*` semantics for drag-to-index.
  - no robust manual-user intent capture for consumable target parameters.
  - shop/pack stochastic parity lacks an explicit sim-vs-oracle audit framework tied to gate output.

## P37 Prioritized Workset (Derived)
- P0:
  - first-class `MOVE_HAND_CARD` and `MOVE_JOKER` actions with `index_base` normalization.
  - first-class `CONSUMABLE_USE` action payload with position-target parameter schema.
  - `SHOP_REROLL`/`SHOP_BUY`/`PACK_OPEN` naming and replay contract normalization.
  - new order-sensitive scope `p37_action_fidelity_core` and directed fixtures.
  - probability parity audit framework and artifact outputs.
- P1:
  - compatibility aliases for legacy `SWAP_*` naming.
  - broaden shop/buy/pack semantics in native sim beyond replay-context injection.
- P2:
  - high-frequency manual drag gesture reconstruction (multi-drag chain) from sparse snapshots.

## Validation Commands (T1)
```powershell
Get-Content -Raw sim/spec/TRACE_CONTRACT.md
Get-Content -Raw sim/spec/action_v1.json
Get-Content -Raw sim/spec/trace_v1.json
Get-Content -Raw trainer/record_real_session.py
Get-Content -Raw trainer/real_executor.py
Get-Content -Raw trainer/real_trace_to_fixture.py
Get-Content -Raw sim/tests/run_real_action_replay_fixture.py
Get-Content -Raw sim/tests/run_real_drift_fixture.py
```

## Result Summary (T1)
- Existing contract and runtime pipeline were audited.
- P37 P0/P1/P2 gaps are identified with concrete fixture plans, including mandatory items:
  - 手牌位置移动（拖拽排序）
  - Joker 位置移动（拖拽排序）
  - “左边变右边”类塔罗位置索引语义
  - shop reroll/pack 概率与权重口径
  - 选牌顺序与出牌顺序复核
  - consumable use 参数形态（`cards/index_base/targets`）
