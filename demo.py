import torch
import torch.nn as nn
import torch.nn.functional as F


class ClothesCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(ClothesCNN, self).__init__()

        # 五层卷积
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)

        # 全连接层
        self.fc = nn.Linear(256, num_classes)


    def forward(self, x):

        # 输入图像
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)

        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)

        x = self.conv5(x)
        x = F.relu(x)

        # 全局平均池化
        x = F.adaptive_avg_pool2d(x,(1,1))

        # 展平
        x = torch.flatten(x,1)

        # 分类输出
        x = self.fc(x)

        return x



# 测试网络
if __name__ == "__main__":

    model = ClothesCNN(num_classes=10)

    # 输入一张衣服图片 (3x224x224)
    x = torch.randn(1,3,224,224)

    y = model(x)

    print("输出分类向量尺寸:",y.shape)

