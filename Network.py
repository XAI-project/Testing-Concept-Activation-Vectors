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

        # Caching layer activations
        self.output1 = None
        self.output2 = None
        self.output3 = None
        self.output4 = None
        self.output5 = None
        self.output6 = None

    def forward(self, input):
        # image dim: 3 x 32 x 32
        output1 = F.relu(self.bn1(self.conv1(input)))
        self.output1 = output1
        # image dim: 12 x 30 x 30
        output2 = F.relu(self.bn2(self.conv2(output1)))
        self.output2 = output2
        # image dim: 12 x 28 x 28
        output3 = self.pool(output2)
        self.output3 = output3
        # image dim: 12 x 14 x 14
        output4 = F.relu(self.bn4(self.conv4(output3)))
        self.output4 = output4
        # image dim: 24 x 12 x 12
        output5 = F.relu(self.bn5(self.conv5(output4)))
        self.output5 = output5
        # image dim: 48 x 10 x 10
        output6 = output5.view(-1, 48*10*10)
        self.output6 = output6
        # image dim: 4800
        output = self.fc1(output6)
        # image dim: 10 (number of possible labels)

        return output
