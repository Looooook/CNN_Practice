# RNN-embedding
import torch

input_size = 4
num_class = 4

batch_size = 1

hidden_size = 8
embedding_size = 10

num_layers = 2
seq_len = 5

idx_char = ['h', 'e', 'l', 'o']

# hello
x_data = [[0, 1, 2, 2, 3]]
# ohlol
y_data = [3, 0, 2, 3, 2]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = torch.nn.Embedding(input_size, embedding_size)  # I --> E

        # change rnncell to rnn ** 此时的rnn输入size = embedding size
        self.rnn = torch.nn.RNN(num_layers=num_layers,
                                input_size=embedding_size,
                                hidden_size=hidden_size,
                                batch_first=True)
        # ?只对最后一维进行
        self.fc = torch.nn.Linear(hidden_size, num_class)  # 8 --> 4

    # inputs or input
    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.embedding(x)  # S,B,I --> S,B,E
        x, _ = self.rnn(x, hidden)  # B,S,H
        # print(x.shape)
        x = self.fc(x)  # B,S,H --.> B,S,C
        print(x.shape)
        return x.view(-1, num_class)


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


def train_epoch(epoch):
    inputs = torch.LongTensor(x_data)
    # print(inputs.shape)
    labels = torch.LongTensor(y_data)

    optimizer.zero_grad()
    out = model(inputs)

    loss = criterion(out, labels)
    loss.backward()

    optimizer.step()
    _, idx = out.max(dim=1)
    idx = idx.data.numpy()
    # print(idx.shape)
    for i in idx:
        print(idx_char[i], end='.')
    print('')
    # another
    print('Predicted:', ''.join(idx_char[x] for x in idx))


if __name__ == '__main__':
    for i in range(15):
        train_epoch(i)
