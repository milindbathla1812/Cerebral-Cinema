import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from config import PAD_ID, BOS_ID, EOS_ID, MAX_LEN


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_parallel(en_path, hi_path, max_samples=None):
    en_lines = read_lines(en_path)
    hi_lines = read_lines(hi_path)
    assert len(en_lines) == len(hi_lines), "English and Hindi files must have same number of lines"
    pairs = list(zip(en_lines, hi_lines))
    if max_samples is not None:
        pairs = pairs[:max_samples]
    return pairs


def load_spm(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp


def encode_sentence(sp, text, max_len=MAX_LEN):
    ids = [BOS_ID] + sp.encode(text, out_type=int)[: max_len - 2] + [EOS_ID]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_sp, tgt_sp, max_len=MAX_LEN):
        self.pairs = pairs
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src_ids = encode_sentence(self.src_sp, src_text, self.max_len)
        tgt_ids = encode_sentence(self.tgt_sp, tgt_text, self.max_len)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)
