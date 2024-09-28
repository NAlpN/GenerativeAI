from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import load_dataset

# Veri Seti yükleme
dataset = load_dataset("imdb")

# Tokenizer ile veri hazırlığı
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(example):
    return tokenizer(example['text'], padding = 'max_length', trunclation = True)

tokenized_datasets = dataset.map(tokenize_data, batched = True)

# Model hazırlığı
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Model eğitimi
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

trainer.train()

# Model değerlendirmesi
trainer.evaluate()

# Model değerlendirmesi
trainer.evaluate()