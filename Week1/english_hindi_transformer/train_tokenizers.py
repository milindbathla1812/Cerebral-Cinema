import sentencepiece as spm
from config import *


def train_spm(input_file, model_prefix, vocab_size):
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<sos>",
        eos_piece="<eos>",
        user_defined_symbols=[],
        train_extremely_large_corpus=True,
    )


def main():
    print("Training English tokenizer on TRAIN ONLY...")
    train_spm(TRAIN_EN, SRC_TOKENIZER_PREFIX, VOCAB_SIZE)

    print("Training Hindi tokenizer on TRAIN ONLY...")
    train_spm(TRAIN_HI, TGT_TOKENIZER_PREFIX, VOCAB_SIZE)

    print("Tokenizers saved inside tokenizers/")


if __name__ == "__main__":
    main()
