import torch

input_size = 4
hidden_size = 4
batch_size = 1

idx_char = ['h', 'e', 'l', 'o']

x_data = [0, 1, 2, 2, 3]
y_data = [3, 0, 2, 3, 2]

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
# print(inputs.shape)
labels = torch.LongTensor(y_data).view(-1, 1)


# print(labels.shape)

class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,
                                        hidden_size=self.hidden_size)

    def forward(self, inputs, hidden):
        hidden = self.rnncell(inputs, hidden)
        return hidden

    # initialize
    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


model = Net(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(15):
    loss = 0

    optimizer.zero_grad()
    hidden = model.init_hidden()

    for input, label in zip(inputs, labels):
        # print(input,label)
        hidden = model(input, hidden)
        # print(hidden.shape,label.shape)
        # print(hidden,label)
        loss += criterion(hidden, label)
        # print(hidden.shape, type(label))
        _, idx = hidden.max(dim=1)
        print(idx_char[idx])

    loss.backward()
    print('------------')
    optimizer.step()
