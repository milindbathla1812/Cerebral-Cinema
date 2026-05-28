from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TOKENIZER_DIR = ROOT_DIR / "tokenizers"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# Dataset files
TRAIN_EN = DATA_DIR / "opus.en-hi-train.en"
TRAIN_HI = DATA_DIR / "opus.en-hi-train.hi"
DEV_EN = DATA_DIR / "opus.en-hi-dev.en"
DEV_HI = DATA_DIR / "opus.en-hi-dev.hi"
TEST_EN = DATA_DIR / "opus.en-hi-test.en"
TEST_HI = DATA_DIR / "opus.en-hi-test.hi"

# SentencePiece tokenizer files
SRC_TOKENIZER_PREFIX = TOKENIZER_DIR / "spm_en"
TGT_TOKENIZER_PREFIX = TOKENIZER_DIR / "spm_hi"
SRC_TOKENIZER_MODEL = str(SRC_TOKENIZER_PREFIX) + ".model"
TGT_TOKENIZER_MODEL = str(TGT_TOKENIZER_PREFIX) + ".model"

# Special token IDs. SentencePiece will use these exact IDs.
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# Training settings
MAX_LEN = 80
VOCAB_SIZE = 8000
MAX_TRAIN_SAMPLES = 100000   # use 20000 for laptop, 100000+ for Kaggle GPU
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 3e-4

# Transformer architecture
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

MODEL_PATH = CHECKPOINT_DIR / "best_transformer.pt"
