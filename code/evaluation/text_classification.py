import os, random, argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel

from .eval_static import *
from .eval_bert import *

# set random number
seed_value = 42  
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value) 


def read_file(file_path):
    """Reads text data from a file.
    
    :param file_path: Path to the label file.
    :return: List of strings for each document in corpus.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.readlines()


def read_labels(file_path):
    """Read and preprocess labels from a file.

    :param file_path: Path to the label file.
    :return: Preprocessed list of labels.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        labels = pd.Series([label for line in file for label in line.strip().split()])
        labels = labels.replace('NA', pd.NA)
        labels = labels.astype('category').cat.codes
    return labels.tolist()


def k_fold_split(dataset, k=5):
    """Split dataset into k folds for cross-validation.

    :param dataset: The dataset to split.
    :param k: Number of folds, defaults to 5.
    :return: List of tuples containing train and validation datasets for each fold.
    """
    labels = dataset.labels
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    
    for train_idx, val_idx in skf.split(range(len(dataset)), labels):
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        folds.append((train_dataset, val_dataset))
    
    return folds


class SequenceClassifier(nn.Module):
    """LSTM-based sequence classifier."""
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(SequenceClassifier, self).__init__()
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        lstm_output_dim = hidden_dim * 2
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, embeddings):
        # embeddings: [batch_size, seq_length, embedding_dim]
        lstm_out, _ = self.lstm(embeddings)  # [batch_size, seq_length, lstm_output_dim]
        lstm_out = self.dropout(lstm_out)
        final = lstm_out[:, -1, :]  # [batch_size, lstm_output_dim]
        logits = self.fc(final)  # [batch_size, num_classes]
        
        return logits


def main_eval_loop(model_path, is_bert, corpus_path, label_dir, output_path, tokenizer_path=None, input_label=None, batch_size=8, epoch=10):
    """Perform evaluation of embeddings and models across multiple metadata labels.
    :param model_path: Path to the pre-trained model (BERT or FastText).
    :param is_bert: Flag indicating whether a BERT model is used.
    :param corpus_path: Path to the text corpus file.
    :param label_dir: Directory containing label files corresponding to the corpus.
    :param outputpath: Path to the output file.
    :param tokenizer_path: Path to the BERT tokenizer, required if is_bert is True.
    :param input_label: Specific label file name to use for evaluation, defaults to evaluating all labels in label_dir.
    :param batch_size: Batch size in training, defaults to be 8.
    :param epoch: Number of epochs to run in training, defaults to be 10.
    """
    # ---- load data and label ----
    if input_label:
        label_paths = [(input_label, os.path.join(label_dir, input_label))]
    else:
        label_paths = [(file, os.path.join(label_dir, file)) for file in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, file))]

    try:
        corpus = read_file(corpus_path)
    except FileNotFoundError:
        raise ValueError("Corpus file not found at the specified path.")

    # ---- load model ----
    print("load model ...")
    if is_bert:  # Load BERT model
        try:
            model = BertModel.from_pretrained(model_path)
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path or model_path)
            embed_dim = model.config.hidden_size 
            dataset_cls = BertDataset
            train_fn = train_bert
            eval_fn = evaluate_bert
        except Exception as e:
            raise ValueError(f"Failed to load BERT model and tokenizer: {e}")
    else:  # Load FastText model
        try:
            model = KeyedVectors.load_word2vec_format(model_path, binary=False)
            embed_dim = model.vector_size
            dataset_cls = StaticDataset
            train_fn = train_static
            eval_fn = evaluate_static
        except Exception as e:
            raise ValueError(f"Failed to load FastText model: {e}")
    print(f"load the model from: {model_path}")

    # ---- train and evaluate ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # output_path = os.path.join(output_dir, "extrinsic_eval.txt")

    with open(output_path, "a") as f:
        f.write(f"Evaluation for model: {model_path}\n")
        f.flush()
        for file_name, label_path in label_paths:
            try:
                labels = torch.tensor(read_labels(label_path), dtype=torch.long)
            except FileNotFoundError:
                print(f"Metadata label file not found: {label_path}")
                continue
            num_classes = len(torch.unique(labels))
            dataset = dataset_cls(corpus, labels, model, tokenizer if is_bert else None)

            # 5 fold cross validation
            metrics = {'acc': [], 'prec': [], 'recall': [], 'f1': []}
            folds = k_fold_split(dataset, k=5)
            for i, (train_dataset, val_dataset) in enumerate(folds):
                print(f"evaluate metadata: {file_name}, fold {i + 1}/5, label proportion: {sum(train_dataset.labels) / len(train_dataset.labels)}")
                classifier = SequenceClassifier(embed_dim, 256, num_classes).to(device)
                train_fn(classifier, train_dataset, batch_size=batch_size, epoch_num=epoch, learning_rate=0.001, device=device, bert_model=model if is_bert else None)
                acc, prec, recall, f1 = eval_fn(classifier, val_dataset, 8, device, bert_model=model if is_bert else None)
                metrics['acc'].append(acc)
                metrics['prec'].append(prec)
                metrics['recall'].append(recall)
                metrics['f1'].append(f1)

            # write evaluation results for each label
            f.write(f"\nResults for {file_name}:\n")
            f.flush()
            for metric, values in metrics.items():
                values_str = ", ".join(f"{v:.4f}" for v in values) 
                f.write(f"{metric.capitalize()} - Values: [{values_str}], Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}\n")
            f.write("-" * 50 + "\n")
            f.flush()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance.")

    parser.add_argument("-m", "--model_path", required=True, help="Model file path.")
    parser.add_argument("-b", "--is_bert", action='store_true', help="Use BERT model.")
    parser.add_argument("-c", "--corpus_path", required=True, help="Path to the corpus file.")
    parser.add_argument("-ld", "--label_dir", required=True, help="Directory containing label files.")
    parser.add_argument("-o", "--output_path", required=True, help="Output path.")
    parser.add_argument("-t", "--tokenizer_path", help="Tokenizer path (required for BERT).")
    parser.add_argument("-l", "--input_label", help="Specific label file name.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size in training.")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epoch to run in training.")

    args = parser.parse_args()

    # if use BERT model, one must specify tokenizer path
    if args.is_bert and not args.tokenizer_path:
        parser.error("--is_bert requires --tokenizer_path.")

    main_eval_loop(
        model_path=args.model_path,
        is_bert=args.is_bert,
        corpus_path=args.corpus_path,
        label_dir=args.label_dir,
        output_path=args.output_path,
        tokenizer_path=args.tokenizer_path,
        input_label=args.input_label,
        batch_size=args.batch,
        epoch=args.epoch
    )

if __name__ == "__main__":
    main()


