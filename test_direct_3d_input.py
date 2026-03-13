import torch
import torch.nn as nn

# 模拟MLPResNet的关键部分
class TestMLPResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.SiLU()
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.layer_norm1(x)
        print(f"After LayerNorm1: {x.shape}")
        x = self.fc1(x)
        print(f"After FC1: {x.shape}")
        x = self.act(x)
        x = self.layer_norm2(x)
        print(f"After LayerNorm2: {x.shape}")
        x = self.fc2(x)
        print(f"Final output: {x.shape}")
        return x

# 测试
B, H, D = 4, 7, 512
Z = torch.randn(B, H, D)

model = TestMLPResNet(input_dim=512, hidden_dim=512, output_dim=1)

print("=== 测试直接输入3D张量 ===")
try:
    result = model(Z)
    print(f"SUCCESS: 直接输入3D可以工作！输出形状: {result.shape}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n=== 测试reshape后输入 ===")  
Z_flat = Z.reshape(B * H, D)
try:
    result_flat = model(Z_flat)
    result_reshaped = result_flat.view(B, H, 1)
    print(f"SUCCESS: Reshape方法也工作！输出形状: {result_reshaped.shape}")
except Exception as e:
    print(f"FAILED: {e}")
