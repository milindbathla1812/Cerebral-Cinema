from datasets import load_dataset
ds = load_dataset("Helsinki-NLP/opus-100", "en-hi")



# write to files so your existing code works unchanged
with open('/content/opus.en-hi-train.en', 'w') as f:
    for item in ds['train']:
        f.write(item['translation']['en'] + '\n')

with open('/content/opus.en-hi-train.hi', 'w') as f:
    for item in ds['train']:
        f.write(item['translation']['hi'] + '\n')

import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="/content/opus.en-hi-train.en,/content/opus.en-hi-train.hi",  # BOTH files together
    model_prefix="shared_tokenizer",
    vocab_size=32000,           # larger since covering two languages
    character_coverage=0.9999,  # for Hindi Devanagari
    model_type="bpe",
    shuffle_input_sentence=True,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)

import torch
import torch.nn as nn
import re
from copy import deepcopy
import math
from tqdm import tqdm
######################## VOCABULARY ###########################
vocab_use=['<unk>','<s>','</s>','<pad>']

# creating embedding layer
vocab = {}
with open("shared_tokenizer.vocab", "r", encoding="utf-8") as f:
    for line in f:
        word, idx = line.strip().split()
        if word not in vocab_use:
            idx=idx[1:]
        vocab[word] = int(idx)



for keys , values in vocab.items():
    if values:
        vocab[keys]=values+4
# {'<pad>': 0, '<unk>': 1, 'the': 4, 'cat': 6, ...}
vocab['<unk>']=1
vocab['<s>']=2
vocab['</s>']=3
print(list(vocab.items())[:20])
vocab['▁क']=4

MAX_LEN = 50

with open('opus.en-hi-train.en', 'r', encoding='utf-8') as en_f, \
     open('opus.en-hi-train.hi', 'r', encoding='utf-8') as hi_f, \
     open('opus.en-hi-train-clean.en', 'w', encoding='utf-8') as en_out, \
     open('opus.en-hi-train-clean.hi', 'w', encoding='utf-8') as hi_out:

    total = 0
    kept = 0

    for en_line, hi_line in zip(en_f, hi_f):
        total += 1
        en_line = en_line.strip()
        hi_line = hi_line.strip()

        if len(en_line.split()) <= MAX_LEN and len(hi_line.split()) <= MAX_LEN:
            en_out.write(en_line + '\n')
            hi_out.write(hi_line + '\n')
            kept += 1

print(f"Total pairs  : {total}")
print(f"Kept pairs   : {kept}")
print(f"Removed      : {total - kept}")
print(f"Kept %       : {kept/total*100:.1f}%")

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("shared_tokenizer.model")

# SP already knows all IDs internally
# no need for vocab dict at all

sp = spm.SentencePieceProcessor()
sp.load("shared_tokenizer.model")

def sentence_to_tkns(sentence, vocab):
    sentence = sentence.lower()
    pieces = sp.encode_as_pieces(sentence)  # splits into ['▁what', '▁is', '▁your', '▁name', '?']
    return [vocab.get(piece, 1) for piece in pieces] 

print(sentence_to_tkns("What is your name?",vocab=vocab))


# print(vocab)

###############################################################

##Embedding
embedding=nn.Embedding(32000,512)


# print(output.shape)  # (4, 512) → 4 tokens, each with 512 dimensional vector

from torch.cuda.amp import GradScaler, autocast


def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)  # separate src and trg

    src_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s) for s in src_batch],
        batch_first=True,
        padding_value=0
    )
    trg_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in trg_batch],
        batch_first=True,
        padding_value=0
    )
    return src_padded, trg_padded



test = sentence_to_tkns("is what your name?", vocab)
print(test)

def my_scaled_dot_product_attention(query, key=None, value=None):
    key = key if key is not None else query
    value = value if value is not None else query
    # query and key must have same embedding dimension
    assert query.size(-1) == key.size(-1)

    dk = key.size(-1) # embed dimension of key
    # query, key, value = (bs, seq_len, embed_dim)

    # compute dot-product to obtain pairwise "similarity" and scale it
    qk = query @ key.transpose(-1, -2) / dk**0.5

    # apply softmax
    # attn_weights = (bs, seq_len, seq_len)
    attn_weights = torch.softmax(qk, dim=-1)

    # compute weighted sum of value vectors
    # attn = (bs, seq_len, embed_dim)
    attn = attn_weights @ value
    return attn, attn_weights

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_path, trg_path, vocab, max_len):
        self.src_sn_ls = []
        self.trg_sn_ls = []

        with open(src_path, "r", encoding="utf-8") as sf, \
             open(trg_path, "r", encoding="utf-8") as tf:
            for src_line, trg_line in zip(sf, tf):
                src_line = src_line.strip()
                trg_line = trg_line.strip()
                if src_line and trg_line:
                    src_ids = [2] + sentence_to_tkns(src_line, vocab) + [3]
                    trg_ids = [2] + sentence_to_tkns(trg_line, vocab) + [3]
                    if len(src_ids) <= max_len and len(trg_ids) <= max_len:
                        self.src_sn_ls.append(src_ids)
                        self.trg_sn_ls.append(trg_ids)

    def __getitem__(self, index):
        return self.src_sn_ls[index], self.trg_sn_ls[index]

    def __len__(self):
        return len(self.src_sn_ls)

class EncoderLayer(torch.nn.Module):

    def __init__(self, embed_dim: int, n_heads: int, dim_feedforward: int = 128, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.mha = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True, bias=True)
        self.layer_norm1 = torch.nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm2 = torch.nn.LayerNorm(normalized_shape=embed_dim)

        # section 5.4
        # apply dropout to output of each sublayer before it is added to sublayer's input
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout2 = torch.nn.Dropout(p=dropout)

        # section 3.3 in paper
        self.position_wise_ff = torch.nn.Sequential(
            torch.nn.Linear(in_features=embed_dim, out_features=dim_feedforward, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=dim_feedforward, out_features=embed_dim, bias=True)
        )
    def forward(self, x, src_key_padding_mask=None, src_mask=None):
        # x.shape = (batch_size, seq_len, embed_dim)
        # src_key_padding_mask = (bs, seq_len), True value indicates it should not attend
        # src_mask.shape = (bs, seq_len, seq_len) of dtype torch.bool, True value indicates it shouldn't attend
        attn_output, attn_weights = self.mha(x, x, x, key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        # dropout and residual connection
        x  = x + self.dropout1(attn_output)
        x = self.layer_norm1(x)

        projection = self.position_wise_ff(x)
        # dropout and residual connection
        x = x + self.dropout2(projection)
        # layer norm
        x = self.layer_norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # deepcopy so each layer has independent weights
        self.layers = nn.ModuleList([
            deepcopy(encoder_layer) for _ in range(num_layers)
        ])
    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x      # (B, S, embed_dim) → this is "memory"

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, dim_feedforward=512, dropout=0.1):
        super().__init__()
        # sublayer 1 — masked self attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        # sublayer 2 — cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        # sublayer 3 — feedforward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        # sublayer 1 — self attention with causal mask
        a1, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        x = self.norm1(x + self.dropout1(a1))

        # sublayer 2 — cross attention, Q from decoder, K/V from encoder
        a2, _ = self.cross_attn(
            x, memory, memory,
            key_padding_mask=memory_key_padding_mask
        )
        x = self.norm2(x + self.dropout2(a2))

        # sublayer 3 — feedforward
        x = self.norm3(x + self.dropout3(self.ff(x)))
        return x

class PositionalEncoding(torch.nn.Module):
    # source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#Positional-encoding
    def __init__(self, embed_dim, max_len=50,dropout=0.1):
        super().__init__()
        # create a matrix of [seq_len, hidden_dim] representing positional encoding for each token in sequence
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            deepcopy(decoder_layer) for _ in range(num_layers)
        ])

    def forward(self, x, memory,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        for layer in self.layers:
            x = layer(
                x, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        return x      # (B, T, embed_dim)

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,       # size of shared or source vocab
        tgt_vocab_size,   # size of target vocab
        embed_dim=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=50,
        pad_idx=0
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed_dim = embed_dim

        # --- Embeddings ---
        self.src_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoding  = PositionalEncoding(embed_dim, max_len)

        # --- Encoder stack ---
        encoder_layer = EncoderLayer(embed_dim, n_heads, dim_feedforward, dropout)
        self.encoder  = Encoder(encoder_layer, num_encoder_layers)

        # --- Decoder stack ---
        decoder_layer = DecoderLayer(embed_dim, n_heads, dim_feedforward, dropout)
        self.decoder  = Decoder(decoder_layer, num_decoder_layers)

        # --- Final projection to vocab ---
        self.output_projection = nn.Linear(embed_dim, tgt_vocab_size)

        # --- Weight initialization ---
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_masks(self, src, tgt):
        # src: (B, S)   tgt: (B, T)
        T = tgt.size(1)

        # 1. source padding mask (B, S)
        src_pad_mask = (src == self.pad_idx)

        # 2. target padding mask (B, T)
        tgt_pad_mask = (tgt == self.pad_idx)

        # 3. causal mask (T, T) — upper triangle is True (blocked)
        tgt_causal_mask = torch.triu(
            torch.ones(T, T, device=src.device), diagonal=1
        ).bool()

        return src_pad_mask, tgt_pad_mask, tgt_causal_mask

    def encode(self, src, src_pad_mask):
        # src: (B, S) → embed → encode → memory (B, S, D)
        x = self.src_embedding(src) * math.sqrt(self.embed_dim)
        x = self.pos_encoding(x)
        return self.encoder(x, src_key_padding_mask=src_pad_mask)

    def decode(self, tgt, memory, tgt_causal_mask, tgt_pad_mask, src_pad_mask):
        # tgt: (B, T) → embed → decode → (B, T, D)
        x = self.tgt_embedding(tgt) * math.sqrt(self.embed_dim)
        x = self.pos_encoding(x)
        return self.decoder(
            x, memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

    def forward(self, src, tgt):
        # src: (B, S)   tgt: (B, T)
        # tgt here is trg_in (already shifted right, </s> removed)

        src_pad_mask, tgt_pad_mask, tgt_causal_mask = self.make_masks(src, tgt)

        memory      = self.encode(src, src_pad_mask)
        decoder_out = self.decode(tgt, memory, tgt_causal_mask, tgt_pad_mask, src_pad_mask)

        # project to vocab logits
        logits = self.output_projection(decoder_out)  # (B, T, vocab_size)
        return logits


#################Train###################


if __name__ == "__main__":

    tokenised_dataset=TranslationDataset('opus.en-hi-train-clean.en','opus.en-hi-train-clean.hi',vocab,50)
    # en_tokenised_dataset=tokenised_dataset.src_sn_ls()
    # hi_tokenised_dataset=tokenised_dataset.trg_sn_ls()
    print(len(tokenised_dataset))
    # dataloader =torch.utils.data.DataLoader(tokenised_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    dataloader = torch.utils.data.DataLoader(
        tokenised_dataset,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        batch_size=128,
        drop_last=False  # keeps last incomplete batch
    )

    # ── Hyperparameters ──────────────────────
    VOCAB_SIZE     = 32000
    EMBED_DIM      = 512
    N_HEADS        = 8
    N_LAYERS       = 6
    DIM_FF         = 2048
    DROPOUT        = 0.1
    PAD_IDX        = 0
    EPOCHS         = 10
    LR             = 1e-4

    # ── Model, optimizer, loss ───────────────
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,   # shared vocab for en-hi
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        num_encoder_layers=N_LAYERS,
        num_decoder_layers=N_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=DROPOUT,
        pad_idx=PAD_IDX
    )
    # model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)



    # ── Early Stopping ───────────────────────
    class EarlyStopping:
      def __init__(self, patience=3, min_delta=0.001):
          self.patience = patience      # how many epochs to wait
          self.min_delta = min_delta    # minimum improvement to count
          self.best_loss = float('inf')
          self.counter = 0
          self.stop = False

      def __call__(self, val_loss, model):
          if val_loss < self.best_loss - self.min_delta:
              self.best_loss = val_loss
              self.counter = 0
              # save best model weights
              torch.save(model.state_dict(), "best_model.pt")
              print(f"  ✓ New best loss: {val_loss:.4f} — model saved")
          else:
              self.counter += 1
              print(f"  ✗ No improvement ({self.counter}/{self.patience})")
              if self.counter >= self.patience:
                  self.stop = True
    # ── Training ─────────────────────────────
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    scaler = GradScaler()
    for epoch in range(EPOCHS):
        # ── Training ─────────────────────────────
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

        # for src, trg in loop:
        #     src = src.to(device)
        #     trg = trg.to(device)
        #     trg_in  = trg[:, :-1]
        #     trg_exp = trg[:, 1:]

        #     logits = model(src, trg_in)
        #     loss = criterion(
        #         logits.reshape(-1, VOCAB_SIZE),
        #         trg_exp.reshape(-1)
        #     )


        for src, trg in loop:
            src = src.to(device)
            trg = trg.to(device)
            trg_in  = trg[:, :-1]
            trg_exp = trg[:, 1:]
            optimizer.zero_grad()
            with autocast():
                logits = model(src, trg_in)
                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    trg_exp.reshape(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)         # ✅ handles step internally
            scaler.update()
            # ✅ NO optimizer.step() here

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # ── Early stopping check ──────────────────
        early_stopping(avg_loss, model)
        if early_stopping.stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # load best weights after training
    model.load_state_dict(torch.load("best_model.pt"))
    print("Best model loaded!")
