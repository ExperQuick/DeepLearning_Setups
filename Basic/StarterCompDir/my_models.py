from plf.utils import Component


from torch import nn


class Model1st(Component, nn.Module):
    def __init__(self):
        Component.__init__(self)
        nn.Module.__init__(self)
        self.args = {'conv_deep', 'dense_deep'}

    def _setup(self, args):

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        convs = [ ]
        for i in range(args['conv_deep']):
            convs.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            convs.append(nn.ReLU())

        self.conv3 = nn.Sequential(*convs)


        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        denses = []
        for i in range(args['dense_deep']):
            denses.append(nn.Linear(64 , 64) )
            denses.append(nn.ReLU())
        self.fc2 = nn.Sequential(*denses)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = self.relu(self.conv3(x))

        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x