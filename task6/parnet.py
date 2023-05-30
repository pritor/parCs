import torch
import struct

input_size = 32 * 32

w1_size = 32 * 32 * 16 * 16
w2_size = 16 * 16 * 4 * 4
w3_size = 4 * 4

b1_size = 16 * 16
b2_size = 4 * 4
b3_size = 1

input_buff = []
w1_buff, w2_buff, w3_buff = [], [], []
b1_buff, b2_buff, b3_buff = [], [], []

with open('./input', 'rb') as f:
    input_buff = struct.unpack('f' * input_size, f.read(4 * input_size))

with open('./weights1', 'rb') as f:
    w1_buff = struct.unpack('f' * w1_size, f.read(4 * w1_size))

with open('./weights2', 'rb') as f:
    w2_buff = struct.unpack('f' * w2_size, f.read(4 * w2_size))

with open('./weights3', 'rb') as f:
    w3_buff = struct.unpack('f' * w3_size, f.read(4 * w3_size))

with open('./bias1', 'rb') as f:
    b1_buff = struct.unpack('f' * b1_size, f.read(4 * b1_size))

with open('./bias2', 'rb') as f:
    b2_buff = struct.unpack('f' * b2_size, f.read(4 * b2_size))

with open('./bias3', 'rb') as f:
    b3_buff = struct.unpack('f' * b3_size, f.read(4 * b3_size))


w1_tensor= torch.Tensor(w1_buff).reshape((16 * 16, 32 * 32))
print(w1_tensor.shape)

w2_tensor = torch.Tensor(w2_buff).reshape((4 * 4, 16 * 16))
print(w2_tensor.shape)

w3_tensor = torch.Tensor(w3_buff).reshape((1, 4 * 4))
print(w3_tensor.shape)

b1_tensor= torch.Tensor(b1_buff).reshape((16 * 16))
print(b1_tensor.shape)

b2_tensor = torch.Tensor(b2_buff).reshape((4 * 4))
print(b2_tensor.shape)

b3_tensor = torch.Tensor(b3_buff).reshape(1)
print(b3_tensor.shape)

class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = torch.nn.Linear(32 * 32, 16 * 16)
    self.fc2 = torch.nn.Linear(16 * 16, 4 * 4)
    self.fc3 = torch.nn.Linear(4 * 4, 1)

    self.fc1.weight = torch.nn.Parameter(w1_tensor, requires_grad=True)
    self.fc2.weight = torch.nn.Parameter(w2_tensor, requires_grad=True)
    self.fc3.weight = torch.nn.Parameter(w3_tensor, requires_grad=True)

    self.fc1.bias = torch.nn.Parameter(b1_tensor, requires_grad=True)
    self.fc2.bias = torch.nn.Parameter(b2_tensor, requires_grad=True)
    self.fc3.bias = torch.nn.Parameter(b3_tensor, requires_grad=True)

    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, input):
    x = input
    x = self.sigmoid(self.fc1(x))
    x = self.sigmoid(self.fc2(x))
    x = self.sigmoid(self.fc3(x))

    return x

net = Net()

input = torch.Tensor(input_buff).reshape((1, 1024))
x = net(input)
print(x)


