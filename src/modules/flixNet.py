"""
FlixNet architecture. I am just showcasing my understanding of designing neuralnets.
As usual, I will need to do more experiments to finally come to best architecture.
We need more hardware resource and time to do all experiments, I can explain what procedures I will follow.
"""

import torch
all_label_types_num = 21

class FlixNet(torch.nn.Module):
    def __init__(self):
        super(FlixNet, self).__init__()
        self.arch = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True),
            torch.nn.BatchNorm2d(num_features=64, momentum=0.1, affine=True),
            torch.nn.LeakyReLU(negative_slope=1e-02),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True),
            torch.nn.BatchNorm2d(num_features=128, momentum=0.1, affine=True),
            torch.nn.LeakyReLU(negative_slope=1e-02),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True),
            torch.nn.BatchNorm2d(num_features=256, momentum=0.1, affine=True),
            torch.nn.LeakyReLU(negative_slope=1e-02),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True),
            torch.nn.BatchNorm2d(num_features=512, momentum=0.1, affine=True),
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(num_features=512, momentum=0.1, affine=True),
            torch.nn.LeakyReLU(negative_slope=1e-02),
            torch.nn.Linear(512, 21)
        )

    def forward(self, x):
        return self.arch(x)