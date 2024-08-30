import torch
import torch.nn as nn
import torch.optim as optim

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.transformer = nn.Transformer(
            embed_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout=dropout
        )
        
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        
        transformer_out = self.transformer(src_emb, tgt_emb)
        
        output = self.fc(transformer_out)
        return output

vocab_size = 1000
embed_size = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6

model = SimpleTransformer(vocab_size, embed_size, num_heads, num_encoder_layers, num_decoder_layers)

src = torch.randint(0, vocab_size, (10, 32))
tgt = torch.randint(0, vocab_size, (20, 32))

output = model(src, tgt)

print(output.shape)