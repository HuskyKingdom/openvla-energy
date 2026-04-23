# VEL v2 重设计方案

> Vision-Dominated Energy Learning — 从 "+1% average" 的 marginal gain 变成 reviewer-defensible 的工作。
> 覆盖 current code 的三个致命问题：**saturated sigmoid / scene–action mis-binding 当负样本 / train-inference distribution gap**。

---

## 1. 当前实现的病灶诊断

### 1.1 训练信号远弱于 paper 描述

- `vla-scripts/finetune_Energy_freeze.py:501` 的 active `energy_loss` 只剩 `energy_inbatch_swap_infonce`（batch 内别人家的 expert action 当负样本）。
- `layer_actions` 在 `vla-scripts/finetune_Energy_freeze.py:483-494` 算出来了但没被使用；`compute_negative_energy` 函数整个 dead path。
- Paper Fig. 5 里画的 "Rand / Surface / Final / GT" 四类负样本，实际训练时只用了 "other batch GT"。
- Swap-across-batch 的语义是 **scene–action mis-binding** —— 它让 critic 学会的是 "哪个 trajectory 匹配哪个 scene"，不是 "在这个 scene 下这个 action 好不好"。这是推理时 `a_BC`（L1≈0.05）拿不到 meaningful gradient 的根因。

### 1.2 Energy parameterization 把梯度摁死

`vla-scripts/energy/energy_model.py:243-255`：

```python
energy_feature_step = energy_feature_step * 0.5
E = self.act(energy_feature_step) * self.energy_scale + self.energy_offset  # sigmoid, bounded [0.1, 2.1]
```

- Sigmoid 两端饱和 → `∇_a E` 在真正需要校正的极端点 ≈ 0。
- Energy bounded 在 [0.1, 2.1]，配 `τ=0.5` 时 InfoNCE logits 动态范围 ±4，contrastive 梯度被压。
- Paper Fig. 6 的 "pronounced valley" 在 sigmoid 下本质是 "saturated plateau with shallow dip" —— 正好对应 Fig. 4 里 corrected energy 比原始高的现象 + `k>1 degrade` 的 ablation。

### 1.3 Train-inference distribution gap

- 训练：positive = `a⋆`，negative = 别 batch 的 `a⋆`。
- 推理：对 `a_BC` 做梯度校正；`a_BC` 在 LIBERO demo 状态下 L1(a_BC, a⋆) ≈ 0.05，是训练时从没见过的 "近 expert 点"。
- 结果：critic 在 `a_BC` 邻域 **completely unregularized** —— 训练得再漂亮，真正使用它的工况它没学过。

### 1.4 Gripper 维度污染梯度

`experiments/robot/openvla_utils.py:867-868`：

```python
A = invert_gripper_action_tensor(normalize_gripper_action_tensor(A)).detach().clone().requires_grad_(True)
A[..., -1] = torch.where(A[..., -1] == -1, 1, 0)
```

- Gripper 本质是二值（抓/放），在它上面做连续梯度下降没有物理意义。
- Gripper 维度的 `∇_a E` 常常数值最大 → 整体 step_norm 被抬高触发 clip，挤掉 pose 维度的有效 update。
- 这是 spatial / object / goal 短任务收益 ≈0 但 long 有 +2.2% 的 asymmetry 的来源。

### 1.5 Action chunk mean-pool

- `energy_avg = self.pool(E)` 把 8 步 chunk 的 energy 均等平均。
- "第 1 步错一点" 和 "第 8 步错很多" 给出相同 energy。
- 但 action chunking 真正 commit 的是前几步，前几步 energy 权重必须更大。

### 1.6 Training 动力学 coupling

- Freeze 版本已绕掉了 L1 / energy 双优化器耦合（见 `Energy_L1_Training_Instability_Analysis.md`）。
- 但 `layer_actions` 在 freeze 下是 deterministic 的 → 原本想当 "Surface-like hard negative" 的东西退化成常数，curriculum 价值消失。

---

## 2. Thesis reframe

把叙事从 **"lightweight critic ranks actions"** 改成：

> **"BC policy 在 OOD state 上的失败模式是局部可辨识的 —— 一个只用 vision-dominant feature 的 energy head 能显式建模 BC 的失败 basin，并以 Newton-like 的单步校正逃离。"**

三个直接后果：

1. 训练数据不再是 expert vs 其他 expert，而是 *expert vs BC 会犯的错的一阶近似*。
2. Energy 不需要 bounded —— margin-based ranking 就够，且 gradient 永远非零。
3. Inference 校正从 "fixed-α gradient step" 变成 "line-search / energy-monitored"，kill k>1 degrade。

---

## 3. 新版本设计（6 个模块）

### 3.1 模块 A — Structured Negative Mining

对每个 `(s_t, a⋆_t)`，合成 4 类 BC-like 负样本（无 rollout、无额外数据）：

| 类别 | 生成方式 | 对应 BC 的失败 |
|---|---|---|
| **N1 Temporal** | `a⋆_{t±k}`, k∈{1,2,3} | BC 早 / 晚执行 |
| **N2 Amplitude** | `a⋆_t · s`, s∈{0.5, 1.5, 2.0} | BC overshoot / undershoot |
| **N3 Directional** | `a⋆_t + ε·u`, u 是主分量方向 | BC 某维偏 |
| **N4 Cross-task** | Batch 内其他 expert | scene–action mis-binding（保留） |
| **N5 Gripper-flip** | 翻转 gripper 位 | gripper confusion |

每 positive 配 M=6 个负样本（1 N1 + 2 N2 + 2 N3 + 1 N4，N5 作为 gripper head 的专门训练数据）。InfoNCE 分母从 B → 1+M。

### 3.2 模块 B — Unbounded Energy + Margin Ranking【P1】

去掉 `EnergyModel.forward` 尾部的 sigmoid / scale / offset，E unbounded：

```python
raw = self.prediction_head(Z)          # per-step scalar, unbounded
E   = (raw * weights).sum(dim=1)       # weighted sum (见 3.3)
```

Loss：

```
L = L_nce + λ_gal · L_gal + λ_margin · L_margin
```

- **L_nce**：InfoNCE with M hard negatives，τ=1.0。
- **L_margin**：per-negative margin ranking，`max(0, E(s, a⋆) − E(s, a_neg) + β·L1(a_neg, a⋆))`，β≈2。保证能量差 ∝ L1 距离。
- **L_gal (Gradient Alignment Loss)**：

```
a_tilde = a⋆ + ε,   ε ~ N(0, σ²),   σ ∈ {0.05, 0.15, 0.30}  (multi-scale)
g       = ∇_a E(s, a_tilde)
L_gal   = − E_ε[ cos(−g, a⋆ − a_tilde) ]
```

多尺度 σ 同时覆盖 "BC 附近"（σ=0.05）和 "long-horizon OOD drift"（σ=0.30）。

### 3.3 模块 C — Per-step weighted sum

```python
weights = softmax(linspace(1.0, 0.3, H))   # 前几步权重大
E(s, A) = Σ_t w_t · e_t(s, A)
```

前几步会真被 execute，后面会被 replan 覆盖。

### 3.4 模块 D — Gripper decouple

- 训练：prediction head 分两支，`E_pose`（6 维）和 `E_grip`（1 维）。
- 推理：pose 做 gradient descent；gripper 比较 `E(a_pose, g=0)` vs `E(a_pose, g=1)` 取更低，差距 < δ 则沿用 BC。

### 3.5 模块 E — Energy-monitored Line Search【P1】

替换 `one_step_energy_correction_seq` / `k_step_energy_correction_seq`：

```python
def corrected_action(energy_head, h, a_bc, mask,
                     alphas=(0.2, 0.1, 0.05, 0.0)):
    g     = normalize(∇_a E(h, a_bc, mask))      # 方向单位化
    cands = [a_bc - α·g for α in alphas]         # α=0 保证不劣化
    E_cands = energy_head(h_rep, stack(cands), mask)
    return cands[argmin(E_cands)]
```

- **α=0 always included** → "最坏不比 BC 差"，彻底消灭 k>1 degrade。
- 5 次 energy forward ≈ 0.05ms，忽略不计。

### 3.6 模块 F — In-training Landscape Quality Monitor

每 100 step 在 held-out 小集上算：

1. **Rank-correlation**：N=8 扰动 Spearman(E, ||ε||)，期望 >0.7。
2. **BC→expert alignment**：`cos(−∇_a E(s, a_BC), a⋆ − a_BC)`，期望 >0.5。
3. **Monotonicity violations**：期望 <15%。

log 到 wandb，作为比 `energy_loss` 更可靠的早停信号，也作为 paper 5.2 的量化补充。

---

## 4. 实施 roadmap

| 阶段 | 改动 | 期望 LIBERO-Long gain | 代码量 | 风险 |
|---|---|---|---|---|
| **P1 (1d)** | 3.2 unbounded+margin + 3.5 line-search | +1–2% | ~80 LoC | 低 |
| **P2 (2d)** | 3.1 structured negatives + 3.4 gripper decouple | +1.5–3% | ~200 LoC | 中 |
| **P3 (1d)** | 3.3 weighted sum + 3.6 landscape monitors | +0.5–1% + 论文 ablation | ~60 LoC + logging | 低 |

**期望终态**：Long 94.4 → 97–98，Spatial / Object / Goal +0.5–1%，平均 97.4 → 98.3+。

**稳定性信号**：P1 做完 `energy_loss` 曲线应从震荡 → 单调下降；否则降低 GAL 的 σ ({0.02, 0.08, 0.20})。

---

## 5. Paper-side 改动（最小化）

1. Fig. 6 energy landscape 重跑 —— unbounded E 会变成真正的 bowl valley，视觉升级显著。
2. Table 1 加 "VEL-v2" 行，保留旧 VEL 作 ablation（"w/o structured negatives" / "w/o GAL"）。
3. Sec 4.1.1 末尾加 "negative construction" 段；Sec 4.1.2 residual update 替换成 line-search 描述，notation 不变。

---

## 6. 几个预设的 reviewer 质询

**Q: N1 temporal 负样本是否太容易？**
A: 正因为像才 informative。E 区分 `a_t` vs `a_{t+1}` 必须读 vision 里的任务进展信号 —— 直接对应 VEL 的 vision-dominated 叙事。

**Q: 为什么不用高斯 noise 就够？**
A: 高斯是 isotropic，critic 学到的 gradient 方向也 isotropic；但 BC error 在真实中是 anisotropic 的（overshoot、lag）。N1/N2/N3 是 physically motivated anisotropic negatives。

**Q: π0.5 的 180ms 尴尬怎么办？**
A: 180ms 来自 server-client + JAX round-trip。Line-search 改 batched correction —— 5 个 candidate 堆成 batch 一次发送，server 端一次性算完，预期回到 ~85ms。
