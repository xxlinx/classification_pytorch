import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torch.cuda
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#定义训练的设备
device =torch.device('cuda' if torch.cuda.is_available() else "cpu")

def main():
    #把几个tensor的变化过程打包为一个整体
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),#把导入的PIL或者numpy.ndarray to  tensor
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   #normalize 标准化的一个函数

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=torchvision.transforms.ToTensor())
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=torchvision.transforms.ToTensor())
    # length 长度
    train_set_size = len(train_set)
    val_set_size = len(val_set)

    train_loader = DataLoader(train_set, batch_size=32,
                                               shuffle=True, num_workers=0)  #在num_workers window里只能为0
    val_loader = DataLoader(val_set, batch_size=64,
                                             shuffle=False, num_workers=0)

    # #iter  把val_loader转化为可迭代
    # val_data_iter = iter(val_loader)
    # val_image, val_label = next(val_data_iter)
    
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #导入网络模型
    net = LeNet()
    net = net.to(device)
    #损失函数
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)
    #优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 添加tensorboard
    writer = SummaryWriter("./logs_train")
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0
    # 训练的轮数
    epoch = 20
    for i in range(epoch):  # loop over the dataset multiple times  训练集迭代5次
        print("-------第 {} 轮训练开始-------".format(i + 1))
        running_loss = 0.0  #叠加训练的损失
        net.train()
        for data in train_loader:   #enumerate不仅返回data 还返回步数 start=0，所以step会从0开始

            # get the inputs; data is a list of [inputs, labels]
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()     #历史损失函数梯度清零
            # forward + backward + optimize
            outputs = net(imgs)
            loss = loss_function(outputs, labels)
            loss.backward() #反向传播
            optimizer.step() #参数更新
            total_train_step = total_train_step + 1
            if total_train_step % 100 == 499:

                print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
        # 加入eval() 只对某些层起作用
        net.eval()
        total_test_loss = 0
        # 整体正确的个数
        total_accuracy = 0
        with torch.no_grad():
            for data in val_loader:
                imgs, labels = data
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = net(imgs)
                loss = loss_function(outputs, labels)
                total_test_loss = total_test_loss + loss.item()

                # 分类问题的正确率 argmax（1）表示方向是横向
                accuracy = (outputs.argmax(1) == labels).sum()
                total_accuracy = total_accuracy + accuracy

        print("整体测试集上的Loss: {}".format(total_test_loss))
        print("整体测试集上的正确率: {}".format(total_accuracy / val_set_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / val_set_size, total_test_step)
        total_test_step = total_test_step + 1
    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)
    print('模型已保存')
    writer.close()

if __name__ == '__main__':
    main()
