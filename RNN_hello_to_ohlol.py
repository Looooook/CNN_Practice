# RNN
import torch

input_size = 4
batch_size = 1
hidden_size = 4

# num_layers
num_layers = 1

idx_char = ['h', 'e', 'l', 'o']

# hello
x_data = [0, 1, 2, 2, 3]
# ohlol
y_data = [3, 0, 2, 3, 2]

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
# s b i
inputs = torch.Tensor([one_hot_lookup[x] for x in x_data]).view(num_layers, -1, batch_size, input_size)


class Net(torch.nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, batch_size):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # change rnncell to rnn
        self.rnn = torch.nn.RNN(num_layers=self.num_layers,
                                input_size=self.input_size,
                                hidden_size=self.hidden_size)

    # inputs or input
    def forward(self, inputs, hidden):
        out,hidden = self.rnn(inputs, hidden)
        return out.view(-1,self.hidden_size)

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)


model = Net(num_layers=num_layers,
            input_size=input_size,
            hidden_size=hidden_size,
            batch_size=batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


def train_epoch(epoch):
    hidden = model.init_hidden()
    loss = 0
    # input和层数没有关系
    inputs = torch.Tensor([one_hot_lookup[x] for x in x_data]).view(-1, batch_size, input_size)
    labels = torch.LongTensor(y_data).view(-1)


    optimizer.zero_grad()
    out = model(inputs, hidden)

    loss = criterion(out, labels)
    loss.backward()

    optimizer.step()
    _,idx = out.max(dim=1)
    idx = idx.data.numpy()
    # print(idx.shape)
    for i in idx:
        print(idx_char[i],end='.')
    print('')


if __name__ == '__main__':
    for i in range(15):
        train_epoch(i)