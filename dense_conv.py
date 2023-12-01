import torch
from torch import nn


class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net = torch.nn.Sequential(
            # just like nn.Conv3d but don't support group
            torch.nn.Conv3d(32, 64, 3),
            # non-spatial layers can be used directly in SparseSequential.
            nn.BatchNorm1d(64),
            nn.ReLU(),
            torch.nn.Conv3d(64, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # when use submanifold convolutions, their indices can be shared to save indices generation time.
            torch.nn.Conv3d(64, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            torch.nn.ConvTranspose3d(64, 64, 3, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.shape = shape

    def forward(self, input, features, coors, batch_size):
        # unlike torch, this library only accept int coordinates.
        coors = coors.int()
        return self.net(input)  # .dense()
