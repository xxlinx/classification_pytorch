import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# 准备MNIST训练集
train_dataset = datasets.MNIST(root='./',  # root代表数据存放的路径，“./”就代表与我目前这个文件同级别的位置
                               train=True,  # 将train设置为True就代表我们载入的是训练集的数据
                               transform=transforms.ToTensor(),  # .ToTensor()函数就是将我们加载的图片转换为Tensor的形式
                               download=True)  # 这个代表是否要下载数据集，如果设置为True，当在root中检测不到数据集，系统会自动下载
# 准备MNIST测试集
# 和训练集格式基本相同，差别在于train的参数设置为了False
test_dataset = datasets.MNIST(root='./',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

# 装载训练集数据
train_loader = DataLoader(dataset=train_dataset,  # dataset的名字就是我们刚才准备的的train_dataset
                          batch_size=16,  # 模型批次大小
                          shuffle=True)  # 是否打乱

# 装载训测试集数据
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=16,
                         shuffle=True)

# for i, data in enumerate(train_loader):
#     inputs, labels = data  # data里面有两个值，数据和标签
#     print(inputs.shape)
#     print(labels.shape)
#     break  # 每个批次数据格式都是相同的，所以用break控制只打印一次
#>>> torch.Size([64, 1, 28, 28])
#>>> torch.Size([64])
# 64 代表批次大小。我们上一步设置的是64
# 1 代表通道数 。MNIST数据集是灰度图，所以只有一个通道
# 28 代表图像尺寸。28像素×28像素
# 64 代表64个标签数值，数值范围就是0-9之间，表示我们该批次64张图的标签

# len(train_loader)
# >>>938
# 我们设置的批次大小为64，MINST一共60032张图片，每次拿64张图，一共需要拿938次


# 定义网络结构
class Net(nn.Module):
    def __init__(self):  # 固定格式
        super(Net, self).__init__()  # 固定格式
        self.fc1 = nn.Linear(784, 10)  # 我们简单定义一个784个输入，10个输出的层
        self.softmax = nn.Softmax(dim=1)  # 加一个激活函数，增加模型的非线性能力。这里为什么要限定第1维度呢？
        # 因为我们上层的输出值是（64，10），所以我们要把第1个维度（维度从0开始算）的值做一个概率的转换，
        # 将网络的输出值转化为概率值

    def forward(self, x):  # 固定格式
        # ([64, 1, 28, 28])->(64,784)
        x = x.view(x.size()[0], -1)  # 这部分比较特殊，由于我们设置的Linear的输入是784，但是我们图像的格式是[64, 1, 28, 28]，所以这里我们要做一个转换
        x = self.fc1(x)  # 这里加上我们上面定义的网络
        x = self.softmax(x)
        return x


# 定义模型
my_model = Net()
# 定义损失函数
mse_loss = nn.MSELoss()
# 定义优化器
optimizer = optim.SGD(my_model.parameters(), lr=0.1)  # lr代表学习率


# 模型训练
def train():
    for i, data in enumerate(train_loader):
        # 获得一个批次的数据和标签
        inputs, labels = data
        # 获得模型预测结果（64,10）
        out = my_model(inputs)
        # to onehot,把数据标签变成独热编码
        # (64)-(64,1)
        labels = labels.reshape(-1, 1)
        # tensor.scatter(dim, index, src)
        # dim:对哪个维度进行独热编码
        # index:要将src中对应的值放到tensor的哪个位置。
        # src:插入index的数值
        one_hot = torch.zeros(inputs.shape[0], 10).scatter(1, labels, 1)
        # 计算loss,mes_loss的两个数据的shape要一致
        loss = mse_loss(out, one_hot)
        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 修改权值
        optimizer.step()


# 模型测试
def test():
    correct = 0
    for i, data in enumerate(test_loader):
        # 获得一个批次的数据和标签
        inputs, labels = data
        # 获得模型预测结果（64,10）
        out = my_model(inputs)
        # 获得最大值，以及最大值所在的位置
        _, predicted = torch.max(out, 1)  # 1代表维度。 我们要拿到10个值里面概率最大的值
        # 预测正确的数量
        correct += (predicted == labels).sum()  # 使用64个预测值和64个标签做对比，判断他们是否相等；
        # 也就是判断这64个值里面有多少个预测值和真实标签是相同的
    print("Test acc:{0}".format(correct.item() / len(test_dataset)))


if __name__ == '__main__':
    for epoch in range(30):
        print('epoch:', epoch)
        train()
        test()
