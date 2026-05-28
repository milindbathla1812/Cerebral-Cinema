import torch
from config import *
from dataset import load_spm, encode_sentence
from transformer_model import TransformerTranslationModel


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    device = get_device()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    cfg = checkpoint["config"]
    model = TransformerTranslationModel(**cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    src_sp = load_spm(SRC_TOKENIZER_MODEL)
    tgt_sp = load_spm(TGT_TOKENIZER_MODEL)
    return model, src_sp, tgt_sp, device


def greedy_translate(model, src_sp, tgt_sp, sentence, device, max_len=MAX_LEN):
    src_ids = encode_sentence(src_sp, sentence, max_len)
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

    tgt_ids = [BOS_ID]
    for _ in range(max_len - 1):
        tgt = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(src, tgt)
        next_id = int(logits[:, -1, :].argmax(dim=-1).item())
        if next_id == EOS_ID:
            break
        if next_id != PAD_ID:
            tgt_ids.append(next_id)

    return tgt_sp.decode(tgt_ids[1:])


def main():
    model, src_sp, tgt_sp, device = load_model()
    print("Loaded model on", device)
    print("Type English sentence. Type 'quit' to stop.")
    while True:
        text = input("English: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        print("Hindi:", greedy_translate(model, src_sp, tgt_sp, text, device))


if __name__ == "__main__":
    main()
