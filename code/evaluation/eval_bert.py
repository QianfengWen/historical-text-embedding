from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def bert_collate_fn(batch):
    input_ids, attention_masks, labels = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    labels = torch.stack(labels)

    return input_ids, attention_masks, labels


class BertDataset(Dataset):
    """ PyTorch Dataset for text data."""
    def __init__(self, corpus, labels, model, tokenizer, nan_value=-1):
        self.corpus = corpus
        self.labels = labels
        self.tokenizer = tokenizer
        self.model = model

        # filter out NaN values
        self.corpus, self.labels = zip(*[(text, label) for text, label in zip(corpus, labels) if label != nan_value])
        self.labels = list(self.labels)
        self.corpus = list(self.corpus)


    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sentence = self.corpus[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=False, truncation=True)
        return inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0), label
    

def train_bert(model, dataset, batch_size, epoch_num, learning_rate, device, bert_model):
    """Train a model on a given dataset.

    :param model: Model to train.
    :param dataset: Dataset for training.
    :param batch_size: Batch size for training.
    :param epoch_num: Number of epochs for training.
    :param learning_rate: Learning rate for the optimizer.
    :param device: Device to train on ('cuda' or 'cpu').
    """
    model.to(device)
    bert_model.to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=bert_collate_fn)

    for epoch in range(epoch_num):
        model.train() 
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epoch_num}")

        for input_ids, attention_mask, labels in progress_bar:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device) 
            
            # feature extraction from bert
            with torch.no_grad():
                outputs = bert_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                embeddings = hidden_states[-2] 

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({'loss': f'{total_loss / (progress_bar.last_print_n + 1):.4f}'})

        print(f"Epoch {epoch+1} completed, Average Loss: {total_loss / len(dataloader)}")


def evaluate_bert(model, dataset, batch_size, device, bert_model):
    """Evaluate a model on a given dataset.

    :param model: Model to evaluate.
    :param dataset: Dataset for evaluation.
    :param batch_size: Batch size for evaluation.
    :param device: Device for evaluation ('cuda' or 'cpu').
    :return: Evaluation metrics (accuracy, precision, recall, f1).
    """
    model.eval() 
    model.to(device)
    bert_model.to(device) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=bert_collate_fn)

    all_predictions, all_true_labels = [], []

    with torch.no_grad():         
         for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device) 
            
            # extract features from bert
            outputs = bert_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            embeddings = hidden_states[-2] 

            outputs = model(embeddings)
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