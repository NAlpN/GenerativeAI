import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

df = pd.read_csv('tweets.csv')

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
        text = text.lower()
    else:
        text = ''
    return text

df['text'] = df['text'].apply(clean_text)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 128

def tokenize_data(texts, tokenizer, max_len):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

input_ids, attention_masks = tokenize_data(df['text'].values, tokenizer, MAX_LEN)

label_dict = {'positive': 1, 'negative': 0, 'neutral': 2}
labels = df['sentiment'].apply(lambda x: label_dict[x]).values

train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, test_size=0.1, random_state=42)
train_masks, test_masks, _, _ = train_test_split(attention_masks, labels, test_size=0.1, random_state=42)

train_lengths = (train_inputs != tokenizer.pad_token_id).sum(dim=1)
test_lengths = (test_inputs != tokenizer.pad_token_id).sum(dim=1)

train_data = TensorDataset(train_inputs, train_masks, train_lengths, torch.tensor(train_labels))
test_data = TensorDataset(test_inputs, test_masks, test_lengths, torch.tensor(test_labels))

train_dataloader = DataLoader(
    train_data,
    sampler=RandomSampler(train_data),
    batch_size=32
)

test_dataloader = DataLoader(
    test_data,
    sampler=SequentialSampler(test_data),
    batch_size=32
)

print("Veri yükleyiciler hazır.")

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(SentimentRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        
        text_lengths = text_lengths.cpu().int()
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        
        return self.fc(hidden)

vocab_size = len(tokenizer.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 3
n_layers = 2
bidirectional = True
dropout = 0.5

model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)

print("Model parametreleri ayarlandı.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        optimizer.zero_grad()
        
        text, text_masks, text_lengths, labels = batch
        text, text_masks, text_lengths, labels = text.to(device), text_masks.to(device), text_lengths.to(device), labels.to(device)
        
        predictions = model(text, text_lengths)
        
        loss = criterion(predictions, labels)
        acc = (predictions.argmax(dim=1) == labels).sum().item() / len(labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

EPOCHS = 5

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
    
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'Eğitim Kayıp Oranı: %{train_loss:.2f} | Eğitim Doğruluk Oranı: %{train_acc:.2f}')

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in iterator:
            text, text_masks, text_lengths, labels = batch
            text, text_masks, text_lengths, labels = text.to(device), text_masks.to(device), text_lengths.to(device), labels.to(device)
            
            predictions = model(text, text_lengths)
            
            loss = criterion(predictions, labels)
            acc = (predictions.argmax(dim=1) == labels).sum().item() / len(labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

test_loss, test_acc = evaluate(model, test_dataloader, criterion)
print(f'Test Kayıp Oranı: {test_loss:.2f} | Test Doğruluk Oranı: {test_acc:.2f}')