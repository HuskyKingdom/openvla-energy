# VEL 改进方案总结

## 背景：原方法的问题

原始 VEL 在 LIBERO 上平均提升约 0.7%，部分 suite 甚至略有下降。根因分析如下：

| 问题 | 表现 | 位置 |
|------|------|------|
| **弱负样本** | In-batch swap 的负样本是"别的 state 下的 expert action"，不是真正的坏动作，contrast 信号弱 | `energy_inbatch_swap_infonce` |
| **Sigmoid 梯度饱和** | `E = sigmoid(...) * 2.0 + 0.1`，能量范围 [0.1, 2.1]，两端梯度趋零，校正信号消失 | `EnergyModel.forward` |
| **训练-推理 gap** | 训练只见 expert actions 和 batch-swap，推理时面对 BC 预测（偏差动作），energy head 在该区域未被优化 | 整体设计 |
| **能量景观非凸** | Table 4 已证：k>1 步校正反而降性能，说明梯度方向不可信，单步校正的有效性缺乏保证 | `k_step_energy_correction_seq` |

---

## 改动一览

### 1. `vla-scripts/energy/energy_model.py`

#### 1.1 `EnergyModel.forward` — Sigmoid → Softplus

**原来：**
```python
energy_feature_step = energy_feature_step * 0.5
E = self.act(energy_feature_step) * self.energy_scale + self.energy_offset
# E ∈ [0.1, 2.1]，梯度在饱和区趋零
```

**现在：**
```python
raw = self.prediction_head(Z)               # [B,H,1]
E = F.softplus(raw, beta=1.0) + 1e-4       # 无上界，梯度处处非零
```

**为什么：** Softplus 是严格单调的非负函数，不会饱和，确保任何动作处的梯度都具有有效幅值，推理时的梯度校正始终有信号。同时移除了 `energy_scale`/`energy_offset` 等硬编码超参数。

---

#### 1.2 新增 `gradient_alignment_loss()` — 梯度对齐损失（GAL）

```python
def gradient_alignment_loss(energy_model, h, a_expert, pad_mask,
                             sigma=0.15, n_samples=4):
    """
    L_align = 1 - cos_sim( ∇_a E(h, a_noisy),  a_expert - a_noisy )
    """
```

**核心思想：** 不仅要求 energy head 能对动作排序，还要求其梯度场**指向 expert action 方向**。

训练时对 expert action 加噪声 `a_noisy = a* + N(0, σ)`，然后：
1. 计算能量梯度 `g = ∇_a E(h, a_noisy)`
2. 计算目标方向 `d = a* - a_noisy`（从噪声点指向 expert）
3. 最大化两者的 cosine similarity：`L_align = 1 - cos_sim(g, d)`

**为什么能解决 "k>1 步校正反而降性能" 的问题：** 当梯度方向被显式训练为指向 expert 时，推理时任意近邻点的单步梯度下降都是可靠的，多步也不会走偏。

**Paper Claim：** *"GAL explicitly regularizes the gradient field of the energy function, ensuring gradient descent from any perturbed action converges toward expert-like behavior."*

---

#### 1.3 新增 `multi_scale_hard_negative_infonce()` — 多尺度困难负样本 InfoNCE

```python
def multi_scale_hard_negative_infonce(energy_model, h, a_pos, pad_mask,
                                       layer_actions, sigmas=(0.05, 0.2, 0.5), tau=0.5):
```

**三源负样本：**

| 来源 | 距离环 | 作用 |
|------|--------|------|
| In-batch swap | 语义不匹配 | 全局排序，区分不同任务的动作 |
| Intermediate-layer BC actions | 中等偏差（部分解码） | 覆盖 BC 预测的常见偏差范围，弥补 train-inference gap |
| Gaussian noise σ=0.05 | 极近邻 | 精细化 expert 附近的能量景观 |
| Gaussian noise σ=0.2 | 近邻 | 覆盖典型 BC 误差范围 |
| Gaussian noise σ=0.5 | 中远邻 | 建立全局凸性 |

**Paper Claim：** *"By sampling negatives at multiple distance scales—from fine-grained perturbations to semantically mismatched actions—VEL shapes the energy landscape at all distances from the expert action, ensuring reliable gradient-based correction regardless of the initial BC proposal's deviation."*

---

### 2. `vla-scripts/finetune_Energy_freeze.py`

#### 训练循环：组合新 loss

```python
# 旧：
swap_loss, E_pos_mean, E_neg_mean = energy_inbatch_swap_infonce(...)
energy_loss = swap_loss

# 新：
nce_loss, E_pos_mean, E_neg_mean = multi_scale_hard_negative_infonce(
    energy_model, context_hidden, ground_truth_actions,
    energy_mask, layer_actions, sigmas=(0.05, 0.2, 0.5), tau=0.5,
)
gal_loss = gradient_alignment_loss(
    energy_model, context_hidden, ground_truth_actions,
    energy_mask, sigma=0.15, n_samples=4,
)
lambda_gal = 0.5
energy_loss = nce_loss + lambda_gal * gal_loss
```

新增 W&B 指标：`nce_loss`、`gal_loss`（方便分别监控两个 loss 的收敛曲线）。

---

### 3. `experiments/robot/openvla_utils.py`

#### 推理校正：梯度归一化

```python
# 旧：直接用梯度幅值
step = alpha * grad_A

# 新：先归一化为单位方向，再乘 alpha
grad_norm = grad_A.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-8)
grad_dir  = grad_A / grad_norm.view(1, 1, 1)     # 单位方向
step       = alpha * grad_dir
```

**为什么：** 原来的步长 = `alpha * ||grad||`，梯度幅值大的地方步长大、小的地方步长小，导致不同 timestep 的校正效果不一致。归一化后 alpha 的含义固定为"在动作空间中移动的距离"，更直观也更稳定。

---

## 改动文件汇总

```
vla-scripts/energy/energy_model.py
  - EnergyModel.forward: sigmoid → softplus（行 184-210）
  - 新增 gradient_alignment_loss()（行 498-552）
  - 新增 multi_scale_hard_negative_infonce()（行 555-631）

vla-scripts/finetune_Energy_freeze.py
  - import 新增两个函数（行 69-78）
  - run_forward_pass: 替换 energy loss 计算（行 506-522）
  - 新增 nce_loss/gal_loss 到 metrics 和 recent_metrics（行 573-574, 1163-1164）

experiments/robot/openvla_utils.py
  - k_step_energy_correction_seq: 梯度归一化（行 941-967）
```

---

## 预期实验观察

训练时重点观察：
- `gal_loss` 应稳步下降（趋向 0，意味着 cosine similarity → 1）
- `E_neg / E_pos` gap 应比原方法更大（更强的对比度）
- 原来 k>1 步校正会降性能；GAL 训练后建议再测一组 k=2/3，预期有正向效果

评测时预期：
- LIBERO-Long 提升最显著（这里 BC 误差积累最严重，GAL 修正效果最明显）
- 更重要的是：σ=0.05 的细粒度负样本可以让 energy head 在 BC 预测的小偏差处也有准确排序
