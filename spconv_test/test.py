import torch
import spconv.pytorch as spconv
# your features with shape [N, num_channels]
features = torch.tensor([[1.], [2.]])
# your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
indices = torch.tensor([[0, 0, 0], [0, 0, 1]]).int()
# spatial shape of your sparse tensor, spatial_shape[i] is shape of indices[:, 1 + i].
spatial_shape = [256, 256]
batch_size = 1  # batch size of your sparse tensor.
x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
x_dense_NCHW = x.dense()  # convert sparse tensor to dense NCHW tensor.
print(x_dense_NCHW)
print(x_dense_NCHW.shape)