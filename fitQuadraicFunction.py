import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F 

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# Variable tensor
x, y = torch.autograd.Variable(x), Variable(y)

# plot
#plt.scatter(x.data.numpy(), y.data.numpy())
#plt.show()

class Net(torch.nn.Module):  # from torch.Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # implents __init__
        # type of every layer
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):   # implents Module's forward function
        # forward and output prediction
        x = F.relu(self.hidden(x))      # activate function
        x = self.predict(x)             # output
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # set prameters learning rate
loss_func = torch.nn.MSELoss()      # loss function

for t in range(100):
    prediction = net(x)     # input x to net ,get prediction

    loss = loss_func(prediction, y)     # compute loss

    optimizer.zero_grad()   # clear parameters
    loss.backward()         # backward
    optimizer.step()        # iter and update the parameters

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(.4)