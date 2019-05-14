import torch
from torch import nn

class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.Linear(4096,2),
            nn.Softmax()
        )
    def forward(self,x):
        x = self.conv(x)
        x = x.view(-1,7 * 7 * 512)
        return self.fc(x)