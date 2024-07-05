import os
import string
import unicodedata
import torch
import torch.nn as nn
import random
import time
import math

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def find_files(path):
    return [os.path.join(path, filename) for filename in os.listdir(path)]

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

def load_data(path):
    category_lines = {}
    all_categories = []
    
    for filename in find_files(path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories

category_lines, all_categories = load_data('dataset2')
n_categories = len(all_categories)

def letter_to_index(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def name_to_tensor(name):
    tensor = torch.zeros(len(name), 1, n_letters)
    for i, letter in enumerate(name):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor

def category_to_tensor(category):
    li = all_categories.index(category)
    tensor = torch.tensor([li], dtype = torch.long)
    return tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.view(1, 1, -1)
        output, hidden = self.gru(input, hidden)
        output = self.hidden_to_output(output[0])
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

def random_training_example():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = category_to_tensor(category)
    line_tensor = name_to_tensor(line)
    return category, line, category_tensor, line_tensor

criterion = nn.NLLLoss()
learning_rate = 0.005

def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden()
    rnn.zero_grad()

    output = None

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    if output is not None:
        loss = criterion(output, category_tensor)
        loss.backward()

        for p in rnn.parameters():
            p.data.add_(-learning_rate, p.grad.data)

        return output, loss.item()
    else:
        raise ValueError("Çıkış tensörü boş. Lütfen giriş tensörünü kontrol edin.")

n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    
    if iter % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (
            iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))
    
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

def evaluate(line_tensor):
    hidden = rnn.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(name_to_tensor(input_line))
        
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []
        
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dostoevsky')
predict('Smith')
predict('Yamada')