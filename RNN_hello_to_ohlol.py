# RNN cell
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
inputs = torch.Tensor([one_hot_lookup[x] for x in x_data]).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1,1)

class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # change rnncell to rnn
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,
                                        hidden_size=self.hidden_size)

    # inputs or input
    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


model = Net(input_size=input_size,
            hidden_size=hidden_size,
            batch_size=batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


for epoch in range(2):
    # just initialize hidden
    hidden = model.init_hidden()
    loss = 0
    optimizer.zero_grad()
    for input, label in zip(inputs, labels):
        hidden = model(input, hidden)
        print(input.shape)
        loss += criterion(hidden, label)
        print('-')
        _, idx = hidden.max(dim=1)
        print(idx_char[idx])
    print('++++++++')

    loss.backward()

    optimizer.step()

