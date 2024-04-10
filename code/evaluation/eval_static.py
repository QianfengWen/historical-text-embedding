from tqdm import tqdm
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def build_vocab(corpus):
    """Build the vocab from the corpus.
    :param corpus: A list of string representing corpus.
    """
    tokens = Counter(token for text in corpus for token in text.split())
    vocab = {token: idx + 2 for idx, (token, _) in enumerate(tokens.items())}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    return vocab


def get_embeddings(vocab, model):
    """Generate an embeddings matrix for the given vocabulary using the specified word embedding model.
    :param vocab: dict, a mapping from tokens (words) to indices.
    :param model: `gensim.models.keyedvectors.KeyedVectors`, the Gensim word embedding model to use for generate word embeddings.
    :return: A 2D tensor containing the embeddings for the vocabulary.
    """
    num_tokens = len(vocab)
    emb_size = model.vector_size
    embeddings_matrix = np.zeros((num_tokens, emb_size))

    for word, idx in vocab.items():
        if word in model:
            embeddings_matrix[idx] = model[word]

    return torch.tensor(embeddings_matrix, dtype=torch.float)


def collate_fn(batch):
    """Custom collation function for batching data.
    """
    inputs, labels = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return padded_inputs, labels


class StaticDataset(Dataset):
    """ PyTorch Dataset for text data."""
    def __init__(self, corpus, labels, model, tokenizer=None, nan_value=-1):
        self.corpus = corpus
        self.labels = labels
        self.model = model

        # filter out NaN values
        self.corpus, self.labels = zip(*[(text, label) for text, label in zip(corpus, labels) if label != nan_value])
        self.labels = list(self.labels)
        self.corpus = list(self.corpus)

        self.vocab = build_vocab(self.corpus)
        embeddings = get_embeddings(vocab=self.vocab, model=self.model)
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=0)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        text = self.corpus[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        embedding = sentence_to_embedding(text, self.embedding_layer, self.vocab)
        return embedding, label


###### Extract Features for Word Embeddings ######
def sentence_to_embedding(sentence, embedding_layer, vocab, max_seq_length=256):
    """Generate embeddings for a sentence using a specified FastText model.

    :param sentence: Sentence to embed.
    :param embedding_layer: torch.nn.Embedding, the PyTorch embedding layer used to convert indices into embeddings.
    :param vocab: dict, a mapping from tokens (words) to indices.
    :param max_seq_length: Maximum length of the sentence for padding/truncation.
    :return: Embeddings for the sentence.
    """
    # convert sentence to indices with padding
    idx = [vocab.get(word, vocab["<unk>"]) for word in sentence.split()[:max_seq_length]]
    idx_tensor = torch.tensor(idx, dtype=torch.long)

    # apply the embedding
    embeddings = embedding_layer(idx_tensor)
    return embeddings


def train_static(model, dataset, batch_size, epoch_num, learning_rate, device, bert_model=None):
    """Train a model on a given dataset.

    :param model: Classifier model to train.
    :param dataset: Dataset for training.
    :param batch_size: Batch size for training.
    :param epoch_num: Number of epochs for training.
    :param learning_rate: Learning rate for the optimizer.
    :param device: Device to train on ('cuda' or 'cpu').
    :param bert_model: Bert model as the feature extractor, default is None, required for `train_bert` method only.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epoch_num):
        model.train() 
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epoch_num}")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device) 
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({'loss': f'{total_loss / (progress_bar.last_print_n + 1):.4f}'})

        print(f"Epoch {epoch+1} completed, Average Loss: {total_loss / len(dataloader)}")


def evaluate_static(model, dataset, batch_size, device, bert_model=None):
    """Evaluate a model on a given dataset.

    :param model: Classifier model to evaluate.
    :param dataset: Dataset for evaluation.
    :param batch_size: Batch size for evaluation.
    :param device: Device for evaluation ('cuda' or 'cpu').
    :param bert_model: Bert model as the feature extractor, default is None, required for `train_bert` method only.
    :return: Evaluation metrics (accuracy, precision, recall, f1).
    """
    model.eval() 
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_predictions, all_true_labels = [], []

    with torch.no_grad():         
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.detach().cpu().numpy())
            all_true_labels.extend(labels.detach().cpu().numpy())

    # calculate evaluation metrics
    num_classes = len(np.unique(all_true_labels))
    avg_type = "binary" if num_classes == 2 else "macro"
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average=avg_type, zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average=avg_type)
    f1 = f1_score(all_true_labels, all_predictions, average=avg_type, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return accuracy, precision, recall, f1
