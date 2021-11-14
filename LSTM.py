import torch

idx_char = ['h', 'e', 'l', 'o']

seq_len = 2
batch_size = 3
num_layer = 2
hidden_size = 4
input_size = 4
# hello
x_data = [0, 1, 2, 2, 3, 3]
# ohlol
y_data = [3, 0, 2, 3, 2, 3]

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot_data = torch.Tensor([one_hot_lookup[x] for x in x_data]).view(-1, batch_size, input_size)
y_labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer)
        # self.fc = torch.nn.Linear(24,6)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


net = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for i in range(25):
    output = net(x_one_hot_data).view(6, 4)

    optimizer.zero_grad()
    # print(output.shape, y_labels.shape)
    loss = criterion(output, y_labels)
    loss.backward()

    optimizer.step()
    _, idx = output.max(dim=1)
    print('Epoch:%d' % (i + 1), 'Predicted:', ''.join(idx_char[x] for x in idx))
    print(loss.item())
