import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import pickle
import time
# prepare dataset
import os

batch_size = 100
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.1307, std=0.3081)
])
train_datasets = datasets.CIFAR10(root='../dataset/cifar-10/',
                                  train=True,
                                  download=True,
                                  transform=transform)

train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

test_datasets = datasets.CIFAR10(root='../dataset/cifar-10/',
                                 train=False,
                                 download=True,
                                 transform=transform
                                 )
test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=False)


# 不改变除C以外的参数 输出通道88
class Inception(torch.nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()

        self.conv_ap = torch.nn.Conv2d(in_channels, 24, kernel_size=(1, 1))

        self.conv1x1_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))

        self.conv5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))
        self.conv5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=(5, 5), padding=2)

        self.conv3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))
        self.conv3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=(3, 3), padding=1)
        self.conv3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        conv_ap = F.avg_pool2d(
            self.conv_ap(x), kernel_size=(
                3, 3), padding=1, stride=1)

        conv_1x1_1 = self.conv1x1_1(x)

        conv5x5 = self.conv5x5_1(x)
        conv5x5 = self.conv5x5_2(conv5x5)

        conv3x3 = self.conv3x3_1(x)
        conv3x3 = self.conv3x3_2(conv3x3)
        conv3x3 = self.conv3x3_3(conv3x3)
        # print(conv_ap.shape, conv_1x1_1.shape, conv5x5.shape, conv3x3.shape)
        concat = [conv_ap, conv_1x1_1, conv5x5, conv3x3]

        return torch.cat(concat, dim=1)


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels,
                                     kernel_size=(3, 3), padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels,
                                     kernel_size=(3, 3), padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 30, kernel_size=(5, 5))  # 100 30 28 28
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2, 2))  # 100 30 14 14
        self.incep1 = Inception(in_channels=30)  # 100 88 14 14
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=(5, 5))  # 100 20 10 10
        self.incep2 = Inception(in_channels=20)  # 100 88 10 10
        self.res1 = ResidualBlock(88)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=(2, 2))  # 100 88 5 5

        self.fc1 = torch.nn.Linear(2200, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.incep1(self.mp1(x)))
        x = F.relu(self.incep2(self.conv2(x)))
        x = self.res1(x)
        x = self.mp2(x)
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# dr = Cifar10DataReader(cifar_folder="../dataset/cifar-10-batches-py/")
# d, l = dr.next_test_data()


model = Net().cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


def train_one_epoch(epoch):
    total_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        # 前馈
        inputs, targets = data
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 300 == 299:
            print(
                'EPOCH:{} , LOSS:{},BATCH_IDX'.format(
                    epoch + 1,
                    total_loss / 300,
                    batch_idx + 1))


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images).cuda()
            # output --> max, max_indices
            _, predicted = torch.max(outputs.data, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('accuracy:%d %%' % (100 * correct / total))


if __name__ == "__main__":
    for epoch in range(30):
        tic = time.time()
        train_one_epoch(epoch)
        test()
        toc = time.time()
        t = toc - tic
        print('%.4f' % t)
