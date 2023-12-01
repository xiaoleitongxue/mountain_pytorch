import torch
from torch import nn
import spconv.pytorch as spconv
channels = 3
height = 4
width = 4
# your features with shape [N, num_channels]
input_tensor = torch.zeros(channels, height, width)

input_tensor[:, 0:2, :] = 10
print(input_tensor)
features = input_tensor[:, 0:2, :]

print(features)
features = torch.flatten(features, start_dim=0)

print(features)
features = features.reshape(features.shape[0], 1)
print(features)
none_zero_indexes = torch.nonzero(input_tensor[:, :, :])
# your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
indices = torch.cat(
    (torch.zeros(none_zero_indexes.shape[0], 1), none_zero_indexes), dim=1)

print(indices.shape)

# spatial shape of your sparse tensor, spatial_shape[i] is shape of indices[:, 1 + i].
spatial_shape = [channels, height, width]
# batch size of your sparse tensor.
batch_size = 1
x = spconv.SparseConvTensor(
    features, indices.int(), spatial_shape, batch_size)
print(x)
x_dense_NCHW = x.dense()  # convert sparse tensor to dense NCHW tensor.
print(x_dense_NCHW)
print(x_dense_NCHW.shape)

conv_func = spconv.SparseConv2d(in_channels=1, out_channels=1, kernel_size=1);
y = conv_func(x)
print(y)
y_dense_NCHW = y.dense()  # convert sparse tensor to dense NCHW tensor.
print(y_dense_NCHW)
print(y_dense_NCHW.shape)
