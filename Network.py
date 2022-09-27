import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(
            in_channels=24, out_channels=48, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48*10*10, 10)

    def forward(self, input):
        # image dim: 3 x 32 x 32
        output = F.relu(self.bn1(self.conv1(input)))
        # image dim: 6 x 30 x 30
        output = F.relu(self.bn2(self.conv2(output)))
        # image dim: 12 x 28 x 28
        output = self.pool(output)
        # image dim: 12 x 14 x 14
        output = F.relu(self.bn4(self.conv4(output)))
        # image dim: 24 x 12 x 12
        output = F.relu(self.bn5(self.conv5(output)))
        # image dim: 24 x 10 x 10
        output = output.view(-1, 48*10*10)
        # image dim: 2400
        output = self.fc1(output)
        # image dim: 10 (number of possible labels)

        return output
