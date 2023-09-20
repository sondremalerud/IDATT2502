import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):

    def __init__(self, encoding_size, emoji_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, emoji_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' ' [0]
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h' [1]
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a' [2]
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 't' [3]
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'r' [4]
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], # 'c'  [5]
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], # 'f'  [6]
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], # 'l'  [7]
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], # 'm'  [8]
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], # 'p'  [9]
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], # 's'  [10]
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], # 'o'  [11]
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.], # 'n'  [12]
]
encoding_size = len(char_encodings)

index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']
index_to_emoji = ['🎩', '🐀', '🐈', '🏢', '👨', '🧢', '👦']
emoji_names = ['hat ', 'rat ', 'cat ', 'flat', 'matt', 'cap ', 'son ' ]



emoji_encodings = [
    [ 1., 0., 0., 0., 0., 0., 0.],  # 'hat'
    [ 0., 1., 0., 0., 0., 0., 0.],  # 'rat'
    [ 0., 0., 1., 0., 0., 0., 0.],  # 'cat'
    [ 0., 0., 0., 1., 0., 0., 0.],  # 'flat'
    [ 0., 0., 0., 0., 1., 0., 0.], # 'matt'
    [ 0., 0., 0., 0., 0., 1., 0.], # 'cap'
    [ 0., 0., 0., 0., 0., 0., 1.], # 'son'
]
emoji_size = len(emoji_encodings)
print(emoji_size)


x_train = torch.tensor([[[char_encodings[1]], [char_encodings[2]], [char_encodings[3]],[char_encodings[0]]],    # 'hat '
                        [[char_encodings[4]], [char_encodings[2]], [char_encodings[3]],[char_encodings[0]]],    # 'rat '
                        [[char_encodings[5]], [char_encodings[2]], [char_encodings[3]],[char_encodings[0]]],    # 'cat '
                        [[char_encodings[6]], [char_encodings[7]], [char_encodings[2]], [char_encodings[3]]],   # 'flat'
                        [[char_encodings[8]], [char_encodings[2]], [char_encodings[3]],[char_encodings[3]]],    # 'matt'
                        [[char_encodings[5]], [char_encodings[2]], [char_encodings[9]],[char_encodings[0]]],    # 'cap '
                        [[char_encodings[10]], [char_encodings[11]], [char_encodings[12]],[char_encodings[0]]]] # 'son '
                        )

y_train = torch.tensor([[emoji_encodings[0], emoji_encodings[0], emoji_encodings[0],emoji_encodings[0]],    # 'hat '
                        [emoji_encodings[1], emoji_encodings[1], emoji_encodings[1],emoji_encodings[1]],    # 'rat '
                       [emoji_encodings[2], emoji_encodings[2], emoji_encodings[2],emoji_encodings[2]],     # 'cat '
                       [emoji_encodings[3], emoji_encodings[3], emoji_encodings[3],emoji_encodings[3]],     # 'flat'
                       [emoji_encodings[4], emoji_encodings[4], emoji_encodings[4], emoji_encodings[4]],    # 'matt'
                       [emoji_encodings[5], emoji_encodings[5], emoji_encodings[5],emoji_encodings[5]],     # 'cap '
                       [emoji_encodings[6], emoji_encodings[6], emoji_encodings[6],emoji_encodings[6]],     # 'son '
                      ])  

#print(y_train)
model = LongShortTermMemoryModel(encoding_size, emoji_size)

# RSMprop fungerer som regel bedre enn gradient descent og Adam til recurrent neural networks
optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    for i in range(emoji_size):
        model.reset()

        #print(x_train[i])
        #print(x_train[i].shape)
        #print(y_train[i])
        #print(y_train[i].shape)
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()

# Predicter alle emojis
for i in range(emoji_size):
  model.reset()
  y = model.f(x_train[i])[-1, :]
  print(emoji_names[i] + ": " + index_to_emoji[y.argmax()])




rats_tensor = torch.tensor([[char_encodings[4]], [char_encodings[2]], [char_encodings[3]],[char_encodings[10]]])
rt_tensor = torch.tensor([[char_encodings[4]], [char_encodings[3]],[char_encodings[0]]])
catt_tensor = torch.tensor([[char_encodings[5]], [char_encodings[2]],[char_encodings[3]],[char_encodings[3]]])

model.reset()
y = model.f(rats_tensor)[-1, :]
print(y.detach().numpy())
print("rats :" + index_to_emoji[y.argmax()])

model.reset()
y = model.f(rt_tensor)[-1, :]
print(y.detach().numpy())
print("rt :" + index_to_emoji[y.argmax()])

model.reset()
y = model.f(catt_tensor)[-1, :]
print(y.detach().numpy())
print("catt :" + index_to_emoji[y.argmax()])
