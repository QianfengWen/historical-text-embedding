import os, json
from tokenizers import BertWordPieceTokenizer
from code.utils import *

def train_tokenizer(input_file, tokenizer_path):
    """
    Trains a BertWordPieceTokenizer on a given corpus and saves the tokenizer model along with its configuration.

    param input_file: Path to the text file or files containing the training corpus. 
    param tokenizer_path: Directory where the trained tokenizer model and its configuration will be saved. 
    """
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
    vocab_size = 32000
    max_length = 512
    truncate_longer_samples = False

    # train tokenizer
    tokenizer = BertWordPieceTokenizer(lowercase=True) 
    tokenizer.train(files=input_file, vocab_size=vocab_size, special_tokens=special_tokens, min_frequency=5)

    if truncate_longer_samples:
        tokenizer.enable_truncation(max_length=max_length)

    tokenizer.save_model(tokenizer_path)

    with open(os.path.join(tokenizer_path, "tokenizer_config.json"), "w") as f:
        tokenizer_cfg = {
            "do_lower_case": True,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "model_max_length": max_length,
            "max_len": max_length,
        }
        json.dump(tokenizer_cfg, f)