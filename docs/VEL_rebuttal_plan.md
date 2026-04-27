# VEL — CVPR 2026 Reject后改进计划

**Submission #631 / OpenReview #37126** — Reject (suggested findings workshop), 4个reviewer意见 + AC meta-review.
**剩余时间**: 3天.

## Reviewer攻击点 → 内部诊断

| # | 攻击点 | 来源 | 内部诊断（看代码确认） |
|---|---|---|---|
| 1 | 能量landscape脆 (α=0.5时long-SR 91.4 vs 96.6) | znPP, AC | `EnergyModel.forward` 用 `sigmoid(Z)*2 + 0.1`，输出限在 [0.1, 2.1]。logit范围被压缩，边界附近 ∇E 饱和，descent方向不可信。 |
| 2 | In-batch swap 制造false negative | znPP | `energy_inbatch_swap_infonce` 把同batch别人的expert action当负样本——相似state下的expert action被推高能量，矛盾目标。 |
| 3 | 缺Best-of-N baseline | znPP | 没实现，只用了gradient correction。 |
| 4 | 只用Gaussian验证landscape | znPP | Fig 6只有Gaussian扰动，没有hard-negative perturb曲线。 |
| 5 | "Vision-Dominated"名不副实 | znPP, AC | 架构上full hidden state(vision+text+action)进cross-attention，没有视觉token隔离。 |
| 6 | 短horizon gain marginal / 只测LIBERO | RpjT, Pn8P, AC | 评测scope确实窄。 |

> 攻击点1 & 2其实是**同一根因**：bounded sigmoid压缩可用logit范围，模型挤在窄区间区分正负，需求过苛 → 与false-negative产生的矛盾梯度叠加 → 鞍点/局部最小变多 → α一大就崩.

## 已完成的代码改动 (2026-04-27)

### 1. Unbound the energy output
**文件**: [vla-scripts/energy/energy_model.py](../vla-scripts/energy/energy_model.py)
- `EnergyModel.__init__` 增加 `bounded_energy: bool = False` 参数（默认False = 新行为）.
- `EnergyModel.forward` 根据 `self.bounded_energy` 分流；`False` 直接用 `prediction_head` 的线性输出.
- `nn.Sigmoid()` / `energy_scale` / `energy_offset` 仅在 `bounded_energy=True` 时实例化, 旧checkpoint加载兼容（这些字段不在 state_dict 里）.

### 2. Fix `energy_infonce_loss` 现有bug + 加 `build_structured_negatives`
**文件**: [vla-scripts/energy/energy_model.py](../vla-scripts/energy/energy_model.py)
- 旧函数 `E_pos, _ = energy_model(h, a_pos, reduce=...)` 是错的（model 单返回值），且 `pad_mask` 没跟着 `h_rep` 一起 `repeat_interleave`. 修复后 logits 维度 `[B, 1+M]`, 正样本 index=0.
- 新函数 `build_structured_negatives(layer_actions, a_pos, ...)` 构造每锚点 M 个结构化负样本，来源:
  1. **Layer-action negatives**: `layer_actions[len//4]`, `layer_actions[len//2]` — 中间层 early-exit 给出的 partial-policy 动作，scene-correct 但 action plausible-but-wrong（hard negative).
  2. **Gaussian perturbation**: `add_gaussian_noise(a_pos, sigma=0.3)`.
  3. **Amplitude scaling**: `a_pos * 0.3`, `a_pos * 1.7`（保方向破幅度）.
  4. **Directional flip**: 单维取反（保幅度破方向）.
  返回 `[B, M, H, Da]`, 默认全开 → M=6.

### 3. Wire 进 training path
**文件**: [vla-scripts/finetune_Energy_freeze.py](../vla-scripts/finetune_Energy_freeze.py)
- `FinetuneConfig` 增加: `bounded_energy=False`, `neg_strategy="inbatch_swap"`, `energy_tau=0.5`, `structured_loss_weight=1.0`.
- `run_forward_pass` 增加同名参数, 并在 `with torch.cuda.amp.autocast(enabled=False):` 内部根据 `neg_strategy ∈ {inbatch_swap, structured, both}` 分流.
- Logging metrics 增加 `L_swap`, `L_struct`, `Structured_Negative_Energy`（按需写入），不会因为某分支未跑而 `.item()` 崩.

### 4. Slurm script 默认改新方案
**文件**: [slurms/energy_training.sh](../slurms/energy_training.sh)
- 默认: `--bounded_energy False`, `--neg_strategy structured`, `--energy_tau 0.5`, `--structured_loss_weight 1.0`.
- `--run_id_note` 改为 `unbounded_structured`（旧的 `vel_gal_multiscale` 标签废弃）.

### 暂未改 / 已知遗留
- 验证path `run_forward_pass(...)` 调用点 (`finetune_Energy_freeze.py:899`) 解包数量不匹配（pre-existing bug，但 `use_val_set=False` 是默认值，不会触发，不在本次scope）.
- 旧函数 `compute_negative_energy` / `energy_inbatch_swap_infonce_2d` / `get_negatives` 保留未删，避免破坏其它实验.
- Inference 端 `k_step_energy_correction_seq` 的 energy non-increase guard 未加（Day 2-3 再上）.

## 三天Plan

### Day 1 — 训练 + Best-of-N baseline 实现
- [ ] **跑训练** (`sbatch slurms/energy_training.sh`)：默认就是 `neg_strategy=structured` + `bounded_energy=False`. ~36h 上限, 50K step.
  - 关键监控: `L_struct` 应该比 baseline `L_swap` 高 1-2 个数量级（结构化负样本更难），训练后 `Positive_Energy < Structured_Negative_Energy` 且差距 stable 增长.
  - Failsafe: 若 NaN, 先 `--bounded_energy True` 跑一组对比, 隔离unbound带来的不稳定.
- [ ] **同时**写 Best-of-N baseline 入口（不需训练）：在 `experiments/robot/openvla_utils.py` 新增 `bestofn_energy_select(energy_head, h, a_bc, N=8, sigma=0.05)`，从 base policy 的 a_bc 加 Gaussian 采 N 个候选, energy 最低的 argmin.
- [ ] 起一组 ablation 训练（`--neg_strategy both`）作为 day-2/3 备选 checkpoint.

### Day 2 — 评测 + Hard-negative landscape图
- [ ] 跑 `auto_eval_energy_k.sh` on 4 个 LIBERO suites, 对每个 ckpt 跑 α ∈ {0.1, 0.5, 1.0} × k ∈ {1, 2, 3, 4}.
  - 重点 **Long-SR 在 α=0.5 时**的稳定性 → 直接打 znPP / AC 的"brittle"批评.
- [ ] 跑 Best-of-N N ∈ {1, 4, 8, 16} 在 LIBERO-Long 和 Goal.
- [ ] 写 hard-negative landscape 可视化脚本：用 `build_structured_negatives` 4种扰动各画一条 energy vs 强度曲线 → 替换原 paper Fig 6（或加新 panel）.
  - 对比 `bounded_energy=True/False` 两个checkpoint, 证明 unbounded 后曲线更陡 + 可分.
- [ ] Inference 端加 energy non-increase guard（一行 if 检查）作为 k-step 鲁棒性保险.

### Day 3 — Writing修复 + Vision-Grounded 正名
- [ ] 全文 `Vision-Dominated` → `Vision-Grounded`.
- [ ] Sec 4.1 typo `Energy Leaning` → `Learning`; 4.1.1/4.1.2 编号补齐; refs [6][7] 去重; abstract `OpenVLA和` 后面补 π0.5 全名; intro `leads to substantial performance` 残句.
- [ ] **Vision-Grounded 实证**: 在 `EnergyModel.forward` 的 cross-attention 里加 `vision_token_mask` 入口，限制 K/V 只来自 vision patches。跑一组 ablation（fix `--bounded_energy False --neg_strategy structured`）:
  - `Full tokens` (默认)
  - `Vision-only K/V`
  - `Vision + Language`
  - `Vision + Proprio`
  报告4列 SR, 让 `Vision-only` 的Long-SR保持不掉 → "vision tokens承载主要feasibility信号" 有架构证据.
- [ ] Method 4.1 加 "Negatives Design" 子节，**主动写 false-negative 问题**，把 `build_structured_negatives` 当 first-class contribution.
- [ ] Limitations: 承认 LIBERO 范围限制, RoboCasa 列入 future work.

## Risk & 兜底

| Risk | 兜底 |
|---|---|
| Unbounded 训练 NaN | 切回 `--bounded_energy True`，至少有 structured negatives 这个 contribution |
| structured neg 改动太大、SR 反而掉 | 切 `--neg_strategy both`（structured + swap 一起算，weight 0.5/0.5），保住 swap 基线性能 |
| Best-of-N ≥ VEL gradient | 文章卖点改写：energy head 是 critic，VEL gradient 和 BoN 是 critic 的两种用法；强调 gradient 在 latency 和 sample-efficiency 上的优势 |
| Vision-only ablation 在 short-horizon 大幅掉点 | 改写为 "Vision-Grounded refinement signal, language-conditioned task selection" 双分工，不强求 vision-only |
| 3 天跑不完 RoboCasa | 不上 RoboCasa；Limitations 老实写, 用 Day 2 的 α-robustness + Day 3 的 token-ablation 凑足 reviewer-defensible 点 |

## 监控指标 (wandb)

- `energy_loss` (主指标，should monotonically decrease)
- `L_swap` / `L_struct` (按 strategy 分支计入)
- `Positive_Energy` (vs `Negative_Energy` / `Structured_Negative_Energy` 的 gap → 越大越好)
- `curr_action_l1_loss` / `next_actions_l1_loss` (L1 不应该被 energy training 影响, 因为backbone freeze)

## Reverting 指南

要回到 paper 提交时的 baseline:
```bash
--bounded_energy True --neg_strategy inbatch_swap --energy_tau 0.5
```
其它一切默认.
