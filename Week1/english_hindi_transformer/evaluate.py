import torch
import sacrebleu
from tqdm import tqdm
from config import *
from dataset import load_parallel, load_spm, encode_sentence
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
    model = TransformerTranslationModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, load_spm(SRC_TOKENIZER_MODEL), load_spm(TGT_TOKENIZER_MODEL), device


def translate_one(model, src_sp, tgt_sp, sentence, device, max_len=MAX_LEN):
    src = torch.tensor(encode_sentence(src_sp, sentence, max_len), dtype=torch.long).unsqueeze(0).to(device)
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
    test_pairs = load_parallel(TEST_EN, TEST_HI, None)

    preds, refs = [], []
    for en, hi in tqdm(test_pairs, desc="testing"):
        pred = translate_one(model, src_sp, tgt_sp, en, device)
        preds.append(pred)
        refs.append(hi)

    bleu = sacrebleu.corpus_bleu(preds, [refs])
    print("BLEU:", bleu.score)

    print("\nSample outputs:")
    for i in range(min(10, len(test_pairs))):
        print("EN:", test_pairs[i][0])
        print("REF:", refs[i])
        print("PRED:", preds[i])
        print("-" * 60)


if __name__ == "__main__":
    main()
