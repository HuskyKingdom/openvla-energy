import torch
import matplotlib.pyplot as plt
import numpy as np

# 分析sigmoid缩放的必要性

def analyze_sigmoid_gradient():
    """分析sigmoid在不同输入范围下的梯度特性"""
    
    # 创建输入范围
    x = torch.linspace(-10, 10, 1000)
    
    # Sigmoid函数
    sigmoid = torch.sigmoid(x)
    
    # Sigmoid梯度: sigmoid(x) * (1 - sigmoid(x))
    sigmoid_grad = sigmoid * (1 - sigmoid)
    
    print("=== Sigmoid特性分析 ===")
    print("输入范围 [-10, 10]:")
    print(f"sigmoid(-10) = {torch.sigmoid(torch.tensor(-10.0)):.6f}")
    print(f"sigmoid(-3) = {torch.sigmoid(torch.tensor(-3.0)):.6f}")
    print(f"sigmoid(0) = {torch.sigmoid(torch.tensor(0.0)):.6f}")
    print(f"sigmoid(3) = {torch.sigmoid(torch.tensor(3.0)):.6f}")
    print(f"sigmoid(10) = {torch.sigmoid(torch.tensor(10.0)):.6f}")
    
    print("\n对应的梯度:")
    print(f"grad(-10) = {torch.sigmoid(torch.tensor(-10.0)) * (1 - torch.sigmoid(torch.tensor(-10.0))):.6f}")
    print(f"grad(-3) = {torch.sigmoid(torch.tensor(-3.0)) * (1 - torch.sigmoid(torch.tensor(-3.0))):.6f}")
    print(f"grad(0) = {torch.sigmoid(torch.tensor(0.0)) * (1 - torch.sigmoid(torch.tensor(0.0))):.6f}")
    print(f"grad(3) = {torch.sigmoid(torch.tensor(3.0)) * (1 - torch.sigmoid(torch.tensor(3.0))):.6f}")
    print(f"grad(10) = {torch.sigmoid(torch.tensor(10.0)) * (1 - torch.sigmoid(torch.tensor(10.0))):.6f}")

def analyze_prediction_head_output():
    """分析prediction_head可能的输出范围"""
    
    print("\n=== Prediction Head输出范围分析 ===")
    
    # 模拟MLPResNet的典型输出
    # 通常神经网络最后一层的输出在训练初期可能在[-2, 2]范围
    # 训练过程中可能扩大到[-5, 5]甚至更大
    
    typical_ranges = [
        ("训练初期", [-2, 2]),
        ("训练中期", [-5, 5]), 
        ("训练后期", [-10, 10])
    ]
    
    for stage, (min_val, max_val) in typical_ranges:
        print(f"\n{stage}: raw ∈ [{min_val}, {max_val}]")
        
        # 不缩放的情况
        raw_min, raw_max = torch.tensor(float(min_val)), torch.tensor(float(max_val))
        sigmoid_min = torch.sigmoid(raw_min)
        sigmoid_max = torch.sigmoid(raw_max) 
        grad_min = sigmoid_min * (1 - sigmoid_min)
        grad_max = sigmoid_max * (1 - sigmoid_max)
        
        print(f"  不缩放: sigmoid({min_val}) = {sigmoid_min:.6f}, grad = {grad_min:.6f}")
        print(f"  不缩放: sigmoid({max_val}) = {sigmoid_max:.6f}, grad = {grad_max:.6f}")
        
        # 缩放0.5的情况
        scaled_min, scaled_max = raw_min * 0.5, raw_max * 0.5
        sigmoid_min_scaled = torch.sigmoid(scaled_min)
        sigmoid_max_scaled = torch.sigmoid(scaled_max)
        grad_min_scaled = sigmoid_min_scaled * (1 - sigmoid_min_scaled)
        grad_max_scaled = sigmoid_max_scaled * (1 - sigmoid_max_scaled)
        
        print(f"  缩放0.5: sigmoid({min_val*0.5}) = {sigmoid_min_scaled:.6f}, grad = {grad_min_scaled:.6f}")
        print(f"  缩放0.5: sigmoid({max_val*0.5}) = {sigmoid_max_scaled:.6f}, grad = {grad_max_scaled:.6f}")
        
        print(f"  梯度改善: {grad_min_scaled/max(grad_min, 1e-10):.2f}x, {grad_max_scaled/max(grad_max, 1e-10):.2f}x")

def why_point_five():
    """解释为什么选择0.5这个缩放因子"""
    
    print("\n=== 为什么选择0.5缩放因子 ===")
    
    # Sigmoid梯度最大值在x=0处，值为0.25
    # 有效梯度区间大约在[-3, 3]，此时梯度 > 0.05
    
    scaling_factors = [0.2, 0.3, 0.5, 0.7, 1.0]
    test_inputs = [-5, -2, 0, 2, 5]  # 典型的raw输出
    
    print("不同缩放因子下的梯度对比 (raw输入 = [-5, -2, 0, 2, 5]):")
    print("缩放因子  |", "  ".join([f"grad({x})" for x in test_inputs]))
    print("-" * 60)
    
    for scale in scaling_factors:
        gradients = []
        for raw in test_inputs:
            scaled = torch.tensor(float(raw)) * scale
            sigmoid_val = torch.sigmoid(scaled)
            grad = sigmoid_val * (1 - sigmoid_val)
            gradients.append(f"{grad:.4f}")
        
        print(f"{scale:8.1f}  | " + "   ".join(gradients))
    
    print("\n分析:")
    print("- 缩放0.5能确保大部分raw值映射到sigmoid的敏感区间[-2.5, 2.5]")
    print("- 在这个区间内，sigmoid梯度 > 0.05，避免梯度消失")
    print("- 0.5是经验上的平衡：不会太保守(0.2)，也不会太激进(0.7+)")

if __name__ == "__main__":
    analyze_sigmoid_gradient()
    analyze_prediction_head_output()  
    why_point_five()
