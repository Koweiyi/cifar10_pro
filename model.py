import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_cifar5(nn.Module):
    def __init__(self):
        super(Net_cifar5, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(1024, 5))

    def forward(self, x):
        x = self.module(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    m = Net_cifar5()
    for name, param in m.named_parameters():
        print(name, param.shape)
