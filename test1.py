import torch
import time

channels = 30
height = 1000
width = 1000
# your features with shape [N, num_channels]
input_tensor = torch.zeros(channels, height, width)

input_tensor[:, 0:32, :] = 10

# With square kernels and equal stride
input = torch.randn(256, 256)
m = torch.nn.Conv2d(30, 30, 3, stride=1)
start_time = time.time()
output = m(input_tensor)
print(output)
print("--- %s s ---" % (time.time() - start_time))

print(output.shape)