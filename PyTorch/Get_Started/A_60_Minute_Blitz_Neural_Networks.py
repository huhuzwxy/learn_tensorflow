import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1 * 32 * 32输入
# 卷积（5 * 5）：6 * 28 * 28
# 池化（2 * 2）：6 * 14 * 14
# 卷积（5 * 5）：16 * 10 * 10
# 池化（2 * 2）：16 * 5 * 5
# 全连接（120层）：
# 全连接（84层）：
# 全连接（10层）：

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # 将数据从[n * c * w * h]转换为[n * m]维，作为下一步全连接的输入
        # x = x.view(-1, self.num_flat_features(x))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
    # 将数据从[n * c * w * h]转换为[n * m]维，作为下一步全连接的输入
        size = x.size()[1: ] #除batch size外的大小
        print('size = ', size)
        num_features = 1
        for s in size:
            print(s)
            num_features = num_features * s
        return num_features

net = Net()
print(net)
#params = list(net.parameters())
#print(params)
input = torch.randn(1, 1, 32, 32)
target = torch.randn(1, 10)
print(target)
criterion = nn.MSELoss()

# 优化函数
optimizer = optim.SGD(net.parameters(), lr = 0.01)
optimizer.zero_grad()
output = net(input)
print(output)
loss = criterion(output, target)
loss.backward()
optimizer.step()
print(output)

