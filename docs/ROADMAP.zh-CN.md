# BalatroAI 路线图

> 语言切换： [简体中文](ROADMAP.zh-CN.md) | [English](ROADMAP.en.md)

## 里程碑树

| 里程碑 | 主题 | 状态 |
|---|---|---|
| P0-P13 | Oracle / sim 基线对齐与真实会话漂移闭环 | 已完成 |
| P22-P27 | 实验编排、seed 治理、campaign / release 运维 | 已完成 |
| P29-P37 | 数据飞轮、动作回放统一、自监督与单动作保真 | 已完成 |
| P38 | 长时统计一致性框架（多 seed 压测 + 聚合对齐） | 已完成 |
| P39 | Policy Arena v1（多策略适配、分桶评估、champion 规则输入） | 已完成 |
| P40 | 闭环改进 v1（回放混合、失败挖掘、arena 门控 promotion 建议） | 已完成 |
| P41 | 闭环改进 v2（lineage、课程策略、切片门控、回归分诊） | 已完成 |
| P42 | RL 候选流水线 v1（env adapter、rollout collector、PPO-lite、闭环接入） | 已完成 |
| P43 | 训练策略重心调整（主线 selfsup+RL，BC/DAgger 降级为 legacy baseline） | 已完成 |
| P44 | 分布式 RL 训练（rollout workers、课程策略、多 seed 门控） | 已完成 |
| P45 | 世界模型 / 潜变量规划 v1（数据集、动力学、不确定性、规划挂钩） | 已完成 |
| P46 | Dyna / imagination loop v1（短 imagined rollout、回放增强、消融门控） | 已完成 |
| P47 | 基于不确定性的模型搜索 v1（候选重排、短视 lookahead、arena 消融） | 已完成 |
| P48 | 自适应混合控制器 v1（按状态在 policy / search / wm-rerank 间路由） | 已完成 |
| P49 | GPU 主线 + CPU rollout / GPU learner + dashboard + readiness guard | 已完成 |
| P50 | 真实 CUDA 拉起、GPU 验证、nightly benchmark profile | 已完成 |
| P51 | checkpoint registry、可恢复 nightly campaign、promotion queue | 已完成 |
| P52 | learned router / guarded meta-controller + arena 消融 + campaign 接入 | 已完成 |
| P53 | 后台执行 + 本地 Ops UI + 窗口监管 | 已完成 |
| P54 | learned router / guarded meta-controller + P22 / dashboard / ops-ui 一等集成 | 已完成 |
| P55 | 配置加载加固 + YAML/JSON sidecar 同步 + P54 夜跑全量验证 | 已完成 |
| P56 | learned-router 校准 + 多 seed benchmark + canary promotion 模式 | 已完成 |
| P57 | 夜间自治协议 + attention queue + morning summary | 已完成 |
| P58 | Windows bootstrap + 可移植性加固 + 环境 doctor | 已完成 |
| P59 | AGENTS 标准化 + 自治迭代入口 | 已完成 |
| P60 | AGENTS 层级刷新 + 自治入口标准化 | 已完成 |
| P61 | 工作流加速 + 验证金字塔 + 延迟认证 | 已完成 |

## 当前重点：P61 之后的质量与耐久性

1. 维持 YAML/JSON sidecar 一致性，每次修改 YAML 后运行 `sync_config_sidecars.ps1`。
2. 扩大 P56 夜跑预算、校准切片覆盖和 canary 评估深度。
3. 加固 P57 / P60 / P61 在 blocked campaign、认证排队和 morning summary 排序上的处理。
4. 提升 P58 跨机器交接体验，减少剩余的 Windows 特定假设。
5. 继续加固 P53 后台模式验证，而不只停留在 smoke。
6. 维持配置来源、readiness guarding、AGENTS 一致性、fast loop、认证队列、dashboard 生成、ops 审计日志和自治决策作为默认安全护栏。

## P60 之后的近期方向

- 扩大 imagined-root 覆盖，不再局限于当前 replay 家族
- 通过辅助损失、rollout-value proxy 和 RL-policy routing，进一步收紧 P42 / P45 / P47 的耦合
- 把真实 CUDA benchmark 从 smoke 规模扩到更长预算
- 在 registry 之上继续加强 checkpoint 去重与保留策略
- 丰富 campaign 重启策略与 attention item 处理流，而不只停留在 latest-run resume
- 扩大 P56 针对 controller collapse、OOD 和罕见高风险切片的校准覆盖
- 提升本地运维流，包括更好的 job history、更安全的排队动作、更强的窗口状态诊断和更清晰的 morning summary 分诊
- 继续加固 bootstrap / doctor 对半配置 Windows 主机的诊断
- 在维持不确定性控制和 arena-first 评估前提下，谨慎扩展固定预算搜索之外的能力
- 在单 GPU 长时间稳定后，再考虑更丰富的 GPU 遥测和多 GPU 支持

## 约束说明

- 真实 / oracle 验证依赖本地 Balatro 和 `balatrobot` 运行时。
- 聚合对齐结果必须始终绑定 seed、配置和版本范围来解读。
- 回放级精确对齐可以与“原生闭式权重公式仍部分未知”并存，但报告必须明确披露这个边界。
