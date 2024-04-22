import argparse
from .pretrain_tokenizer import *
from .bert_embeddings import *

def main(args):
    os.makedirs(args.out, exist_ok=True)

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input corpus file not found at {args.input}")

    # pretrain tokenizer
    if args.train_tokenizer:
        os.makedirs(args.tokenizer, exist_ok=True)
        train_tokenizer(args.input, args.tokenizer)
    
    # further train BERT model
    train_BERT(model_name=args.model, input_file=args.input, tokenizer_path=args.tokenizer, output_dir=args.out, is_pretraining=args.pretrain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt BERT for medieval Latin corpora and extract embeddings.")
    parser.add_argument("-i", "--input", type=str, default="inputs/corpus.txt", help="Path to the corpus file.")
    parser.add_argument("-o", "--out", type=str, default="outputs/", help="Directory to save the adapted model, tokenizer, and embeddings.")
    parser.add_argument("-m", "--model", type=str, default="LuisAVasquez/simple-latin-bert-uncased", help="BERT model to use.")
    parser.add_argument("-t", "--tokenizer", type=str, default="model/pretrained_tokenizer", help="Bert tokenizer path.")
    parser.add_argument("--pretrain", action='store_true', help="Flag to indicate pretraining BERT from scratch.")
    parser.add_argument("--train_tokenizer", action='store_true', help="Flag to indicate tokenizer training")
    
    args = parser.parse_args()
   
    main(args)