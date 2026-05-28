import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from dataset import load_parallel, load_spm, TranslationDataset
from transformer_model import TransformerTranslationModel


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def token_accuracy(logits, targets):
    preds = logits.argmax(dim=-1)
    mask = targets.ne(PAD_ID)
    correct = preds.eq(targets).masked_select(mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_acc, steps = 0.0, 0.0, 0

    loop = tqdm(loader, desc="train" if train else "valid")
    for src, tgt in loop:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        with torch.set_grad_enabled(train):
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        acc = token_accuracy(logits, tgt_expected)
        total_loss += loss.item()
        total_acc += acc
        steps += 1
        loop.set_postfix(loss=loss.item(), acc=acc)

    return total_loss / steps, total_acc / steps


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print("Using device:", device)

    src_sp = load_spm(SRC_TOKENIZER_MODEL)
    tgt_sp = load_spm(TGT_TOKENIZER_MODEL)

    train_pairs = load_parallel(TRAIN_EN, TRAIN_HI, MAX_TRAIN_SAMPLES)
    dev_pairs = load_parallel(DEV_EN, DEV_HI, None)

    train_ds = TranslationDataset(train_pairs, src_sp, tgt_sp, MAX_LEN)
    dev_ds = TranslationDataset(dev_pairs, src_sp, tgt_sp, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    model = TransformerTranslationModel(
        src_vocab_size=src_sp.get_piece_size(),
        tgt_vocab_size=tgt_sp.get_piece_size(),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pad_id=PAD_ID,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, dev_loader, criterion, optimizer, device, train=False)

        print(f"Train loss: {train_loss:.4f} | Train token acc: {train_acc:.4f}")
        print(f"Valid loss: {val_loss:.4f} | Valid token acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "src_tokenizer": SRC_TOKENIZER_MODEL,
                "tgt_tokenizer": TGT_TOKENIZER_MODEL,
                "config": {
                    "src_vocab_size": src_sp.get_piece_size(),
                    "tgt_vocab_size": tgt_sp.get_piece_size(),
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_encoder_layers": NUM_ENCODER_LAYERS,
                    "num_decoder_layers": NUM_DECODER_LAYERS,
                    "dim_feedforward": DIM_FEEDFORWARD,
                    "dropout": DROPOUT,
                    "pad_id": PAD_ID,
                }
            }, MODEL_PATH)
            print("Saved best model:", MODEL_PATH)


if __name__ == "__main__":
    main()
