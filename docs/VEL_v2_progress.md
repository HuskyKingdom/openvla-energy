# VEL v2 — 实施进度日志

配套 [docs/VEL_v2_plan.md](VEL_v2_plan.md)。按 session 时间倒序，最新在上。

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
