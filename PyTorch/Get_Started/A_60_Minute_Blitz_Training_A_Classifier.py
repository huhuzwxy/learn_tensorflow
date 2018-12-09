# 数据处理：
# image常用库：Pillow, OpenCV
# audio常用库：scipy, librosa
# text常用库：Python, Cython, NLTK, SpaCy
# PyTorch中用torchvision， torchvision.datasets为数据集， torch.utils.data.DataLoader为数据转换。

# 步骤：
# 加载CIFAR10，归一化
# 定义卷积神经网络 + 损失函数 + 优化函数
#   3 * 32 * 32输入
#   卷积（5 * 5）：6 * 28 * 28
#   池化（2 * 2）：6 * 14 * 14
#   卷积（5 * 5）：16 * 10 * 10
#   池化（2 * 2）：16 * 5 * 5
#   全连接（120层）：
#   全连接（84层）：
#   全连接（10层）：
# 训练网络
# 测试

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 加载数据集，归一化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)
testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

# 定义损失函数和优化函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# 训练网络
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
    # enumerate(s)将s中的对象列为（索引，对象）的格式
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

# 加载图像
def imshow(img):
    npimg = img.numpy()
    plt.show(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# 测试网络
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('groud truth = ', '' .join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('predicted = ', '' .join('%5s' % classes[predicted[j]] for j in range(4)))



