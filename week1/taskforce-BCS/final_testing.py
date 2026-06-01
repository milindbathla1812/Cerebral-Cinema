import torch
import torch.nn as nn
import math
import re
from copy import deepcopy
import sentencepiece as spm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
nltk.download('punkt')

# ── Paste your class definitions here ──
# (PositionalEncoding, EncoderLayer, Encoder, DecoderLayer, Decoder, Transformer)
# ... same as your inference file ...
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
        return x      


# ── Vocab loading (exactly as training) ─────────────────────
vocab_use = ['<unk>', '<s>', '</s>', '<pad>']
vocab = {}
with open("shared_tokenizer.vocab", "r", encoding="utf-8") as f:
    for line in f:
        word, idx = line.strip().split()
        if word not in vocab_use:
            idx = idx[1:]
        vocab[word] = int(idx)

for keys, values in vocab.items():
    if values:
        vocab[keys] = values + 4

vocab['<unk>'] = 1
vocab['<s>']   = 2
vocab['</s>']  = 3

id_to_word = {v: k for k, v in vocab.items()}

sp = spm.SentencePieceProcessor()
sp.load("shared_tokenizer.model")

def sentence_to_tkns(sentence, vocab):
    sentence = sentence.lower()
    pieces = sp.encode_as_pieces(sentence)
    return [vocab.get(piece, 1) for piece in pieces]

# ── Load model ───────────────────────────────────────────────
VOCAB_SIZE = 32000
PAD_IDX    = 0
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer(
    vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    embed_dim=512,
    n_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    pad_idx=PAD_IDX
)
model.load_state_dict(torch.load("best_model.pt", map_location=device, weights_only=False))
model = model.to(device)
model.eval()
print("Model loaded!")

# ── Collate fn for evaluation batches ────────────────────────
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s) for s in src_batch],
        batch_first=True, padding_value=0
    )
    trg_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in trg_batch],
        batch_first=True, padding_value=0
    )
    return src_padded, trg_padded

# ── Dataset ───────────────────────────────────────────────────
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_path, trg_path, vocab, max_len=50):
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

# ── 1. LOSS on test set ───────────────────────────────────────
def compute_test_loss(src_path, trg_path, batch_size=64):
    dataset    = TranslationDataset(src_path, trg_path, vocab)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    criterion  = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    total_loss  = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, trg in dataloader:
            src    = src.to(device)
            trg    = trg.to(device)
            trg_in  = trg[:, :-1]   # decoder input  (drop </s>)
            trg_exp = trg[:, 1:]    # expected output (drop <s>)

            logits = model(src, trg_in)   # (B, T, V)

            # count non-pad target tokens for per-token loss
            non_pad = (trg_exp != PAD_IDX).sum().item()

            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                trg_exp.reshape(-1)
            )
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

# ── 2. Greedy decode (single sentence) ───────────────────────
def translate(src_ids, max_len=50):
    """Takes a list of token ids (with <s> and </s>), returns predicted token ids."""
    src = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_pad_mask = (src == PAD_IDX)

    memory  = model.encode(src, src_pad_mask)
    tgt_ids = [2]  # <s>

    for _ in range(max_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long).to(device)
        tgt_pad_mask   = (tgt == PAD_IDX)
        T              = tgt.size(1)
        tgt_causal_mask = torch.triu(
            torch.ones(T, T, device=device), diagonal=1
        ).bool()

        out    = model.decode(tgt, memory, tgt_causal_mask, tgt_pad_mask, src_pad_mask)
        logits = model.output_projection(out)
        next_token = logits[0, -1, :].argmax().item()

        if next_token == 3:  # </s>
            break
        tgt_ids.append(next_token)

    return tgt_ids[1:]  # strip <s>

# ── 3. BLEU score on test set ─────────────────────────────────
def compute_bleu(src_path, trg_path, max_samples=None):
    """
    Computes corpus-level BLEU.
    references : list of list of list of tokens  (corpus_bleu format)
    hypotheses : list of list of tokens
    """
    dataset = TranslationDataset(src_path, trg_path, vocab)

    references  = []
    hypotheses  = []
    smoothing   = SmoothingFunction().method1  # handles 0-count n-grams

    limit = max_samples if max_samples else len(dataset)

    for i in range(limit):
        src_ids, trg_ids = dataset[i]

        # reference: strip <s> and </s>
        ref_ids   = [t for t in trg_ids if t not in (2, 3, 0)]
        ref_tokens = [id_to_word.get(t, '<unk>') for t in ref_ids]

        # hypothesis: greedy decode
        with torch.no_grad():
            hyp_ids    = translate(src_ids)
        hyp_tokens = [id_to_word.get(t, '<unk>') for t in hyp_ids]

        references.append([ref_tokens])   # corpus_bleu expects [[ref], ...]
        hypotheses.append(hyp_tokens)

        if (i + 1) % 100 == 0:
            print(f"  Translated {i+1}/{limit} sentences...")

    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothing)
    return bleu * 100   # return as percentage

# ── Run evaluation ────────────────────────────────────────────
if __name__ == "__main__":
    TEST_SRC = "test.en"   # your test English file
    TEST_TRG = "test.hi"   # your test Hindi file

    print("=" * 50)
    print("Computing test loss & perplexity...")
    loss, ppl = compute_test_loss('inputs/opus.en-hi-test.en', 'inputs/opus.en-hi-test.hi')
    print(f"  Test Loss       : {loss:.4f}")
    print(f"  Perplexity      : {ppl:.2f}")

    print("\nComputing BLEU score (this may take a while)...")
    bleu = compute_bleu('inputs/opus.en-hi-test.en', 'inputs/opus.en-hi-test.hi')
    print(f"  Corpus BLEU     : {bleu:.2f}")
    print("=" * 50)