# 一、What is PyTorch?
# 替代Numpy充分使用GPU；深度学习平台（灵活、快）

# Tensors：类似ndarrays，可用于GPU加速
from __future__ import print_function
import torch
import numpy as np

# 创建Tensors
# 随机初始化
x0 = torch.rand(5, 3)
print(x0)
# 初始化全0矩阵，类型为long
x1 = torch.zeros(5, 3, dtype = torch.long)
print(x1)
# 直接用数据创建
x2 = torch.tensor([[1, 2, 3.3], [2, 4.4, 5]])
print(x2)
# 根据已有矩阵创建
x3 = x0.new_zeros(5, 3, dtype = torch.double)
print(x3)
x4 = torch.randn_like(x0, dtype = torch.float)
print(x4)
print(x4.shape)
print(x4.size())

# Tensors操作
# 加
add1 = x0 + x4
print(add1)
add2 = torch.add(x0, x4)
print(add2)
add3 = torch.empty(5, 3)
torch.add(x0, x4, out = add3)
print(add3)
x0.add_(x4)
print(x0)
# risize
print(x1.shape)
y0 = x1.view(15)
print(y0)
y1 = x1.view(-1, 5) # 5表示risize成5列的
print(y1)
# Tensor与Numpy array转换
y2 = y1.numpy()
print(y2)
y3 = np.ones([5, 3])
y4 = torch.from_numpy(y3)
print(y4)

# 二、autograd：自动求导机制


