import torch
import torch.nn as nn
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
import copy

def get_clones(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_layers)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()

        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, d_model) # TODO: 여기서 BPE 적용하려면?
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), n_layers)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()

        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), n_layers)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_len, trg_vocab_len, d_model, n_layers, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab_len, d_model, n_layers, heads, dropout)
        self.decoder = Decoder(trg_vocab_len, d_model, n_layers, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab_len)

    def forward(self, src, trg, src_mask, trg_mask):
        # 특이한 것은 model.encoder(), model.decoder(), model.out() 의 형태로 이거를 직접 꺼내서 쓴다는 것
        # Transformer 의 Encoder 와 Decoder 멤버 펑션을 따로 정의해두지 않았음
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

def get_model(opt, src_vocab_len, trg_vocab_len):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab_len, trg_vocab_len, opt.d_model, opt.n_layers, opt.heads, opt.dropout)

    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights', map_location=opt.device))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    if opt.device == torch.device('cuda'):
        model = model.cuda()

    return model