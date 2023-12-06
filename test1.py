import torch
import time

channels = 3
height = 1000
width = 1000
# your features with shape [N, num_channels]
input_tensor = torch.ones(channels, height, width)


# With square kernels and equal stride
input = torch.ones(height, width)
m = torch.nn.Conv2d(channels, channels, 3).cuda()
m.weight = torch.nn.Parameter(torch.ones(3,3,3,3).cuda())
m.bias = torch.nn.Parameter(torch.ones(3).cuda())
start_time = time.time()
output = m(input_tensor.cuda())
print(output)
print("--- %s s ---" % (time.time() - start_time))

print(output.shape)