import torch
import torch.nn as nn
import torch.nn.functional as F

# define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernal
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6,16,3)

        # an affine operation: y = weight * x + bias
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6 * 6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # if the size is a square, you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1

        for each_size in size:
            num_features *= each_size

        return num_features

net = Net()
print(net)

# you just have to define the `forward` function, and the `backward` function (where gradients are computed) is automatically defined for you using `autograd`. You can use any of the tensor operations in the `forward` function
# the learnable parameters of a model are returned by `net.parameters()`
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's weight

# let's try a random 32x32 input
# note: expected input size of this net (LeNet) is 32x32
# to use this net on the MNIST dataset, please resize the images from the dataset to 32x32
input = torch.randn(1,1,32,32)
out = net(input)
print(out)

# zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1,10))

# loss function

# a loss function takes the (output, target) pair of inputs, and computes a value that estimates how far away the output is from the target
# there are several different loss functions under the nn package
# a simple loss is: `nn.MSELoss` which computes the mean-squared error between the input and the target
output = net(input)
target = torch.randn(10) # a dummy target, for example
target = target.view(1, -1) # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

# if you follow `loss` in the backward direction, using its `.grad_fn` attribute, you will see a graph of computations that looks like this
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d 
#       -> view -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss
# so when we call `loss.backward()`, the whole graph is differentiated with respect to the less, and all tensors in the graph that has `requires_grad = Truw` will have their `.grad` tensor accumulated with the gradient
# for illustration, let us follow a few steps backward
print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU

# backprop

# th backpropogate the error, all we have to do is to `loss.backward()`
# you need to clear the existing gradients though, else gradients will be accumulated to existing gradients
# we shall call `loss.backward()`, and have a look at conv1's bias gradients before and after the backward
net.zero_grad() # zeroes the gradient buffers of all parameters

print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)

loss.backward

print("conv1.bias.grad after backward")
print(net.conv1.bias.grad)

# update the weights

# the simplest update rule used in practice is the stochastic gradient descent (sgd):
# weight = weight - learning_rate * gradient
# we can implement this using simple Python code
learning_rate = 0.01

for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# however, as you use neural networks, you want to use various different update rules such as sgd, nesterov-sgd, adam, rmsprop, etc
# to enable this, we built a small package `torch.optim` that implements all these methods
# using it is very simple
import torch.optim as optim

# create your optimiser
optimiser = optim.SGD(net.parameters(), lr = 0.01)

# in your training loop
optimiser.zero_grad() # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimiser.step() # does the update
