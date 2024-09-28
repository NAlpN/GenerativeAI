from transformers import pipeline, BertTokenizer, BertModel

# Model-Tokenizer yükleme
classifier = pipeline('sentiment-analysis')

# Duygu analizi yapma
s = classifier('Bu filmi çok sevdim!')
print(s)

# Tokenizasyon
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
metin = 'Merhaba, ben Alp.'
tokenized_metin = tokenizer(metin)
print(tokenized_metin)

# Yüklenen modeli kullanma
model = BertModel.from_pretrained('bert-base-uncased')
encoding = tokenizer.encode_plus(
    metin,
    return_tensors = 'pt',
    add_special_tokens = True
)
output = model(**encoding)
print(output)