# VEL v2 — 实施进度日志

配套 [docs/VEL_v2_plan.md](VEL_v2_plan.md)。按 session 时间倒序，最新在上。

---

## 2026-04-27 · 路径 B v1 (monotonic) 失败 → 路径 B v2 (trust-region) 实施

### B v1 (monotonic) eval 结果（spatial）

| 配置 | spatial SR |
|---|---|
| α=0 baseline | 92.6 |
| α=0.2 + skip=False + 无 gate | 83.8 |
| **α=0.2 + skip=False + monotonic gate** | **83.4** |

83.4 vs 83.8 差 0.4%（500 episode 1σ ≈ 1.7%），**在噪声内** → gate 几乎没拒绝任何 correction。

### 失败原因：spurious basin 也是单调的

我把"argmin 在 α_max + 单调下降"当作 valley 判据，但这正好也是 spurious basin 的特征：

- α=0：    E_BC ≈ −30
- α=0.05： E ≈ −32（朝 basin 走，能量下降）
- α=0.1：  E ≈ −35（更深）
- α=0.2：  E ≈ −50（落入 basin）

完美单调 + argmin 在 α_max → 我的 gate 判 PASS → 走进 basin。

判定 spurious 与 valley 用 4 个 sample 的能量序列**信息量不够**，需要换思路。

### 路径 B v2: Trust-Region Ratio

数值优化里的标准做法。用 1st-order Taylor 估计能量下降量：

```
predicted_drop = α · ‖∇E(A_BC)‖₂        (因为 direction = ∇E/‖∇E‖, step = -α·dir)
actual_drop    = E_BC − E_best
ρ              = actual_drop / predicted_drop
```

行为：

| 场景 | predicted | actual | ρ |
|---|---|---|---|
| Real local valley（线性区） | ≈ valley slope | ≈ valley slope | **≈ 1.0** |
| Spurious basin（BC 在基底外） | 小（BC 处梯度温和） | 大（basin 底深） | **≫ 1**（5–100x 常见） |
| Wrong direction / saddle / 平坦区 | 中等（梯度有但方向乱） | 接近 0 或负 | **< ρ_lo** |

接受窗口 `ρ ∈ [0.3, 3.0]`：拒绝两端，只接受"线性近似 ±3 倍"的合理修正。

### 实施

| 文件 | 改动 |
|---|---|
| [experiments/robot/openvla_utils.py:1036-1047](../experiments/robot/openvla_utils.py#L1036-L1047) | `line_search_energy_correction_seq` 新增 `accept_mode='trust'` + `trust_rho_lo/hi` |
| [experiments/robot/openvla_utils.py:1182-1196](../experiments/robot/openvla_utils.py#L1182-L1196) | trust-region ratio 计算 + 接受窗口判定 |
| [experiments/robot/openvla_utils.py:1413-1423](../experiments/robot/openvla_utils.py#L1413-L1423) | call site 增加两个 cfg 字段 |
| [experiments/robot/libero/run_libero_eval.py:155-160](../experiments/robot/libero/run_libero_eval.py#L155-L160) | 新增 `energy_trust_rho_lo / energy_trust_rho_hi` CLI flag，默认改 `accept_mode='trust'` |
| [auto_eval_energy.sh](../auto_eval_energy.sh) | shell 变量 + 4 个 suite 的调用都已 thread |

### B v2 期望行为

- **Spurious basin** （ρ ≫ 3）→ reject → 回退 α=0 = BC，**保底 = baseline**
- **Wrong direction** （ρ < 0.3）→ reject → 同上
- **Real valley** （ρ ≈ 1）→ accept → 应用校正 → SR 提升

最坏情形：rejected rate ~100% → SR ≈ baseline，**至少不破坏 BC**。
最好情形：rejected ~80–90%，剩下 10–20% 都是真 valley → SR > baseline + 1%。

### 决策点（同 v1）

跑完 4 suite 后看：
- SR ≥ baseline + 1% → P1+B v2 是有效方法，写进 paper Table 1
- baseline ± 0.5% → trust gate 太严或 P1 critic 没用，进 Path C (P2)
- SR < baseline → trust 实现有 bug 或 ρ 阈值不对，调 `[ρ_lo, ρ_hi]`

### 调参建议

如果 spatial 跑出来 ≈ baseline（92.6），说明 trust 拒绝率接近 100%。这时可以：
1. 放宽 `ENERGY_TRUST_RHO_HI` 到 5.0 或 10.0（容忍更深的 basin —— 但风险高）
2. 增大 `ENERGY_ALPHA` 到 0.3 或 0.4（让真 valley 更显著，predicted 也大）

如果 spatial 显著低于 baseline（< 90），说明 ρ 公式有 bug，开 `verbose=True` 在 line_search 看实际 ρ 分布。

---

## 2026-04-27 · 路径 A 失败 → 路径 B 实施完成

### 路径 A eval 结果

| Suite | α=0 baseline | α=0.2 **无** skip | α=0.2 **有** skip（Path A） |
|---|---|---|---|
| spatial | 92.6 | 83.8 | **80.0** ↓ |
| object | 98.8 | 41.8 | **36.0** ↓↓ |

skip-gripper **反而让 SR 又掉 3.8 / 5.8 个点**。

### 重新诊断

skip-gripper 之后单位 direction 完全集中在 pose 维度 → pose 维度的实际步幅被放大 → 把原本被 gripper 数值掩盖的 **pose 维度错梯度**完全暴露。结论：

- gripper 不是主犯，只是把症状显形了
- **Pose 维度的能量梯度场也是坏的**
- 根本原因：训练时 NCE/margin/GAL 只约束了 expert 处的能量值和 ε 球内的梯度方向，**球外能量场是任意的、有大量 spurious 局部低点**
- Line-search 的 argmin 设计假设"低能量 = 接近 expert"，**但这个假设只在 expert ε 球内成立**；在球外，argmin 是个**贪婪坑利用器**，把动作往 spurious basin 推

P1 整体设计被证伪：训练目标和 inference SR 不正相关，靠 line-search 在 rugged landscape 上做 argmin 会被坑吃掉所有理论增益。

### 路径 B 实施（已落地）

加 **acceptance gate** 拒绝 spurious basin。原理：

> 真正的 valley 是单调下降的；spurious basin 是 isolated 低点。
> 检查 α-grid 上的能量序列是否**单调非增**，且 argmin 落在 **α_max**（最深步）。
> 任一不满足 → 拒绝校正，返回 α=0 候选（= BC，可证不劣化）。

**代码改动**：

| 文件 | 内容 |
|---|---|
| [experiments/robot/openvla_utils.py:1036-1209](../experiments/robot/openvla_utils.py#L1036-L1209) | `line_search_energy_correction_seq` 新增 `accept_mode / tau / monotonic_tol`；4 种模式 `always / monotonic / slope / both`；reject 时回退到 α=0 candidate |
| [experiments/robot/openvla_utils.py:1393-1414](../experiments/robot/openvla_utils.py#L1393-L1414) | call site 从 `cfg.energy_accept_mode` 等读取 |
| [experiments/robot/libero/run_libero_eval.py:151-156](../experiments/robot/libero/run_libero_eval.py#L151-L156) | 新增 3 个 CLI flag：`energy_accept_mode / energy_tau / energy_monotonic_tol` |
| [auto_eval_energy.sh](../auto_eval_energy.sh) | 加 3 个 shell 变量 + 文档注释 |

**默认值**：
```
ENERGY_ACCEPT_MODE=monotonic    # 拒绝 spurious basin
ENERGY_TAU=4.0                  # slope 模式的 threshold (only used if mode=slope/both)
ENERGY_MONOTONIC_TOL=0.0        # 单调检查的容差
```

### 路径 B 期望 SR

最坏情况：所有校正都被 reject → SR ≡ α=0 baseline（92.6 / 98.8 / 96.4 / 94.6）→ P1 整体不破坏 BC，可写 ablation 行。

中等情况：reject 大部分、accept 少数高质量校正 → SR 略高于 baseline (+0.3~1%) → P1 部分 work，靠 Path B 抢救。

最好情况：reject 比例正好，accept 都是真 valley → SR > baseline + 1%。

### 决策点

跑 Path B 的 4-suite 全套：
- 如果 **接近 baseline ± 0.5%** → P1 推理设计被证伪但不损害 BC，**进 Path C (P2)** 改训练分布
- 如果 **baseline + 0.5–1.5%** → P1 部分成功，可考虑：
  - (a) 调 `ENERGY_TAU` 找更好工作点（小 ablation 价值）
  - (b) 直接 P2，把 P1+B 的数字当 baseline-improvement
- 如果 **baseline + > 1.5%** → P1+B 已是有效 inference 策略，paper Table 1 主行用这个

### 风险与开放问题（持续）

- **Reject 比例需要监测**：开 `verbose=True` 看到 ACCEPT/REJECT 比例。如果 99% reject → gate 太严，调 `monotonic_tol` 放松；如果 0% reject → gate 失效，等于 always 模式。
- **训练目标 vs SR 解耦** 仍未解。P2 必须实现 plan §3.6 landscape monitor，**用 held-out rank-correlation 替代 GAL_cos** 做主要早停指标。
- **能量绝对值漂移**：Run#1/2/3 的 E_pos 各不同，loss 缺 absolute anchor。P2 加 `λ_anchor · E_pos²` (λ≈0.001)，对 slope 模式的 τ 设定关键。
- **GAL_cos = 0.4 天花板**：P2 的 N1–N5 structured negatives 应能突破到 0.55+。如果 P2 后仍 ≤ 0.45 才考虑加 capacity。

---

## 2026-04-27 · P1 eval 结果 → 灾难性退化诊断 → 路径 A/B/C 决策

### Eval 结果

| Suite | α=0（≈BC） | α=0.2（VEL v2 P1） | Δ |
|---|---|---|---|
| spatial | **92.6** | 83.8 | **−8.8** |
| object | **98.8** | 41.8 | **−57.0** ⚠ |
| goal | **96.4** | 77.6 | **−18.8** |
| 10 (long) | **94.6** | 33.0 | **−61.6** ⚠ |

α=0 数字与原 paper Table 1（96.2 / 98.3 / 96.2 / 90.7）大致对得上，证明 **BC 链路健康**。
α=0.2 下能量校正**主动 push 动作变差**，object/long 灾难性。

### 训练曲线（Run#3，50k）

`L_nce ≈ 0.36`、`L_margin ≈ 0.04`、`GAL_cos ≈ 0.39`、E_pos = −32 / E_neg = −9（gap≈23）。
**所有 P1 success criteria 都达成了**，但 SR 反而崩 → 训练目标与 SR 脱钩。

### 根因：gripper 维度污染（异质性是关键证据）

按 "gripper 操作密度" 重排 SR 退化：

| Suite | gripping 事件密度 | Δ |
|---|---|---|
| spatial | 低 | −8.8 |
| object | 高（精确抓取） | **−57.0** |
| goal | 中 | −18.8 |
| long | 极高（链式 pick & place） | **−61.6** |

**Δ 严重程度严格单调对应 gripper 密度**，不是巧合。

物理机制：
1. 能量头训练时 **gripper 取值 ∈ {0, 1}**（expert 二值 + GAL σ≤0.15 的轻微噪声）
2. 推理时 line-search 会跨过这个 ball 边界，gripper 在 (0.2, 0.8) 中段值是**未定义外插区**
3. `g = ∇_a E` 在 gripper 维度往往数值最大（外插行为不规则），**单位化后 gripper 主导 direction**
4. α=0.2 把 gripper 推到 ~0.3–0.5，进入外插区
5. argmin 在外插区**几乎随机选**，落进 spurious basin → 错误 gripper 输出 → 抓取失败

### 三条下一步路径

#### **路径 A · 紧急 patch（半小时，零重训）**

不动模型不动训练，**只在 line-search 推理时把 gripper 维度的 gradient 强制置 0**：

```python
g = ∇_a E(h, a_BC)
g[..., -1] = 0          # 不让 gripper 参与 unit direction
direction = g / (||g|| + ε)
```

效果预测：
- object 41.8 → ~95%+（彻底切断中毒源）
- long 33.0 → ~93%+
- spatial / goal 接近或微高于 α=0 baseline

诊断价值：直接量化 "gripper 是不是真凶"。
- 如果 patch 后 SR 回 baseline 但不超 → P1 实质失败，跳过 B 直接上 C
- 如果回 baseline + 0.5–2% → P1 部分成功，可选 B 优化或进 C

#### **路径 B · 加 acceptance gate（一天，零重训）**

给 line-search 加保守接受准则：

```
E_corrected < E_BC − τ·L1(corrected, BC)
AND  ‖corrected − BC‖_∞ < δ
```

τ ≈ 0.5、δ ≈ 0.05。同时屏蔽 gripper 中毒和 pose 维度上的 spurious basin。
**实验价值**：trust-region size vs SR 是一组好的 ablation 行。

#### **路径 C · 直接 P2 + gripper decouple（3 天，重训）**

对应 plan §3.1 + §3.4：
- **Structured negatives (N1–N5)** 替换 swap → 能量场在 BC-error 方向被显式监督
- **Gripper decouple** → 能量头分两支，gripper 走 argmin{E(g=0), E(g=1)}

预期：LIBERO-Long 94.6 → 96.5+。

### 决策

**先做路径 A**。两条理由：
1. 半小时即可拿到诊断信息，确认/证伪 gripper 假设。结论直接影响 P2 (§3.4) 的优先级。
2. 即使 P1 整体不成功，路径 A 后的数字仍能写进 paper 当 ablation 行（"w/o gripper-skip"）。

### 风险 & Open Questions（持续）

- **训练目标和 SR 的脱钩**是 P2 设计警钟。P2 必须同时实现 plan §3.6 landscape monitor，**用 held-out rank-correlation 替代 GAL_cos** 做主早停指标。
- **能量绝对值漂移**：Run#1 / #2 / #3 的 E_pos 各不同（−4 / +211 / −32），loss 缺 absolute anchor。P2 加 `λ_anchor · E_pos²` (λ≈0.001) 把 E 拉到 0 附近，对 trust-region 阈值设定关键。
- **GAL_cos = 0.4 天花板**仍待 P2 验证。P2 后 GAL_cos > 0.5 是 capacity 不需 scale up 的证据；≤ 0.45 才考虑 §3.x 的多层 cross-attn。

---

## 2026-04-24 · P1 run#1 诊断 → 超参调整

### Run#1 结果（wandb run `vel_gal_multiscale`, 5220 steps）

| 曲线 | 读数 | 诊断 |
|---|---|---|
| L_nce | 稳定 ~0.7（上界 log(8)=2.08） | 判别力 OK ✓ |
| L_margin | 稳定 ~0.1 | Margin 基本满足 ✓ |
| Positive Energy | 收敛到 ~−4 | Unbounded E 正常工作 ✓ |
| Negative Energy | 收敛到 ~+1，gap≈5 | 判别度 OK ✓ |
| **GAL_cos** | **0.2 → 0.06（退步）** | ⚠️ **核心 P1 目标未达成** |
| L_nce/margin spikes @ step 500 | 单次 spike 到 3–4 | Grad clip 兜住，非致命 |

### 根因分析

1. **λ_gal=0.1 太小**
   总 loss ≈ `0.7 (NCE) + 1.0·0.1 (margin) + 0.1·(−0.06) (GAL) = 0.794`
   → GAL 贡献仅 0.8%，优化器几乎不 care。

2. **σ=0.30 太远**
   LIBERO 上实际 BC error L1 ≈ 0.05，σ=0.30 的扰动是 6× off；那里 energy field 平坦，grad 方向就是噪声，稀释了小 σ 的有用信号。

3. **推理端仍会有小增益**
   因为 unbounded E + line-search (α=0 兜底) 是架构改动，与 GAL 独立；保守估计 LIBERO-Long +0.5~1% 来自这两者，GAL 本身本轮没贡献。

### 调整（已落地 [vla-scripts/finetune_Energy_freeze.py:519-526](../vla-scripts/finetune_Energy_freeze.py#L519-L526)）

```python
lambda_gal  = 0.1               → 1.0              # GAL 占比 ~1% → ~40%
gal_sigmas  = (0.05,0.15,0.30)  → (0.03,0.08,0.15) # 对齐 BC-error L1≈0.05
```

### Run#2 判成功 criteria

- `GAL_cos` 5k step 内爬到 **≥0.3**（headline）
- `L_nce` 保持在 0.5–1.0（不因 GAL 权重变大而 blow up）
- `Pos/Neg Energy gap` ≥4（判别度不退）

**Fallback**：`L_nce` 长期 >1.5 或 loss diverge → `lambda_gal` 回降 0.3、σ 回到 `(0.05,0.15,0.30)`。

---

## 2026-04-23 · P1 实施完成 + SDPA 双反传修复

### 已落地的 4 块代码改动

| 模块 | 文件 | 要点 |
|---|---|---|
| **Unbounded E**（训练） | [vla-scripts/energy/energy_model.py:184-273](../vla-scripts/energy/energy_model.py#L184-L273) | 去掉 `sigmoid*2.0+0.1`，直接用 per-step raw scalar → mean pool |
| **Unbounded E**（推理） | [experiments/robot/libero/energy_model/model.py:184-258](../experiments/robot/libero/energy_model/model.py#L184-L258) | 镜像改动（checkpoint shape 兼容） |
| **VEL v2 loss 套装** | [vla-scripts/energy/energy_model.py:540-711](../vla-scripts/energy/energy_model.py#L540-L711) | `energy_margin_swap` / `gradient_alignment_loss` / `vel_v2_energy_loss` |
| **训练脚本接线** | [vla-scripts/finetune_Energy_freeze.py:516-536](../vla-scripts/finetune_Energy_freeze.py#L516-L536) | 替换单项 `swap_loss`；新增 `L_nce / L_margin / L_gal / GAL_cos` 到 wandb |
| **推理 line-search** | [experiments/robot/openvla_utils.py:975-1067](../experiments/robot/openvla_utils.py#L975-L1067) | 1 次 grad + K 候选 batch 评估 + argmin；α=0 always in grid |
| **Call site 切换** | [experiments/robot/openvla_utils.py:1305-1314](../experiments/robot/openvla_utils.py#L1305-L1314) | `cfg.energy_alpha` 当作 α_max；grid = `(α_max, α_max/2, α_max/4, 0)` |

### SDPA 双反传修复

Run#1 第一次跑直接报：

```
RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented
```

根因：`nn.MultiheadAttention` + `batch_first=True` + `need_weights=False` 默认走 flash / memory-efficient SDPA，这两个后端没实现 double-backward。GAL 的 `create_graph=True` 刚好需要。

**Fix**：在 `EnergyModel.forward` 调用 `self.cross(...)` 外围加 `with _math_sdpa():` 强制 MATH 后端（支持双反传；B=8, seq≈300 小 attention 几乎无代价）。

- [vla-scripts/energy/energy_model.py:10-30](../vla-scripts/energy/energy_model.py#L10-L30) · `_math_sdpa()` 定义（新 API 优先，降级 `torch.backends.cuda.sdp_kernel`）
- [vla-scripts/energy/energy_model.py:262-265](../vla-scripts/energy/energy_model.py#L262-L265) · 训练端包裹
- [experiments/robot/libero/energy_model/model.py:9-22](../experiments/robot/libero/energy_model/model.py#L9-L22) + forward · 推理端镜像（训/测 backend 一致避免 bf16 数值差）

---

## 2026-04-23 · 诊断 + Plan 文档落地

### 核心诊断（对齐 code，非 memory 快照）

1. **Active loss 只有 swap InfoNCE** — `layer_actions` / GAL / multi-scale 之前只在 memory 里，code 里没落地
2. **Sigmoid 把 E 夹在 [0.1, 2.1]** — 两端饱和 + 压 InfoNCE logits 动态范围
3. **Swap-across-batch 负样本** — 教的是 scene–action mis-binding，不是 action quality
4. **Train on expert, infer on BC** — L1(a_BC, a⋆)≈0.05 的近邻是 untrained 区域
5. **Gripper 维度被当连续梯度处理** — 物理不 make sense，还污染 pose 维度 grad

### Thesis reframe

从 "lightweight critic ranks actions" 改成：
> **"BC policy 在 OOD state 上的失败模式是局部可辨识的 — 一个 vision-dominant energy head 能显式建模 BC 失败 basin 并以 Newton-like 单步校正逃离。"**

### 产出

- [docs/VEL_v2_plan.md](VEL_v2_plan.md) — 6 模块 / 3 阶段完整设计

### 进度总览

| 阶段 | 内容 | 状态 |
|---|---|---|
| **P1** | Unbounded E + margin + GAL + line-search | ✅ code done · 🔄 第二轮训练调参 |
| **P2** | Structured negatives (N1–N5) + gripper decouple | ⏸ pending |
| **P3** | Per-step weighted sum + landscape quality monitors | ⏸ pending |

---

## 下一步 Action Items

1. **立刻可做**：跑 Run#2 验证 λ_gal=1.0 + σ=(0.03,0.08,0.15)。
2. **Run#2 完成后**：
   - 如果 GAL_cos ≥0.3 → 开跑 LIBERO eval；如果 Long SR ≥95.5 → P1 宣告成功，进入 P2
   - 如果 GAL_cos 仍 <0.2 → 检查是否 GAL 与 NCE 几何冲突，考虑 L2 grad-field supervision（plan §3.2 的 fallback）
3. **P2 可并行准备**：实现 N1–N5 structured negatives 的合成函数，独立于 P1 结果。

## 风险 & Open Questions

- **Step 500 spike**：Run#1 出现两次小 spike（L_nce/L_margin 同时），疑似 batch 里 padding 全 True。grad clip 兜住，但 Run#2 若复现需加 `nan-safe` 分支跳过。
- **GAL 双反传代价**：Run#1 单 step 时延比旧版 +30–50%，可接受；若 Run#2 加大 λ_gal 后训练时间不可接受，考虑把 GAL 改成每 K step 执行一次。
- **Inference端 `energy_head.eval() ↔ train()` 切换**：[openvla_utils.py:1042-1046](../experiments/robot/openvla_utils.py#L1042-L1046) 在 eval 时切到 eval 模式；注意如果有 dropout/batchnorm 行为差异需 audit（current: `PositionalEncoding` 有 dropout=0.2，确实会受影响）。
