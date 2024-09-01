import os, argparse
from tqdm import tqdm

from transformers import BertModel, BertTokenizerFast
import torch
import numpy as np
from code.utils import *


def is_single_model_path(folder_path):
    print(os.listdir(folder_path))
    return any("config.json" in file for file in os.listdir(folder_path))

def save_embeddings(file_path, word_embeddings):
    """
    Save the word embeddings dictionary to a .vec file.

    param file_path: Path to the .vec file to save.
    param word_embeddings: A dictionary with words as keys and embeddings as values.
    """
    # count the number of embeddings and the size of each embedding
    num_embeddings = len(word_embeddings)
    embedding_size = len(next(iter(word_embeddings.values())))

    with open(file_path, "w") as file:
        # write the header with the number of embeddings and their size
        file.write(f"{num_embeddings} {embedding_size}\n")

        # write each word and its embedding
        for word, embedding in word_embeddings.items():
            embedding_str = " ".join(map(str, embedding))
            file.write(f"{word} {embedding_str}\n")


def seq_to_token_embeddings(corpus, model, tokenizer, vocab, device):
    """
    Computes and returns the average token embeddings for each unique word in a corpus.

    :param corpus: A list of sentences to process.
    :param model: The model used to compute token embeddings, expected to return hidden states.
    :param tokenizer: The tokenizer corresponding to 'model' used for tokenizing sentences.
    :param vocab: A set of vocab.
    :param device: The device on which the model computations are performed ('cuda' or 'cpu').
    :return: A dictionary mapping each unique word to its average embedding vector.
    """
    word_embeddings_sum = {}

    with torch.no_grad():
        for sentence in tqdm(corpus, desc="Processing sentences"):
            inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, return_offsets_mapping=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            offset_mapping = inputs.pop('offset_mapping').cpu().numpy()[0]
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            embeddings = torch.mean(torch.stack(hidden_states[-4:]), dim=0).squeeze(0)
            embeddings = embeddings.to('cpu').numpy()

            word = ""
            word_embedding = []
            for i, (start, end) in enumerate(offset_mapping):
                if i + 1 < len(offset_mapping):
                    next_start = offset_mapping[i + 1][0]
                else:
                    next_start = None  # Last token in the sequence

                if start == end:  # Skip special tokens
                    continue
                if next_start is not None and end == next_start:  # Word not finished
                    word += sentence[start:end].lower()
                    word_embedding.append(embeddings[i])
                else:  # Word ends
                    word += sentence[start:end].lower()
                    word_embedding.append(embeddings[i])
                    word_embedding = np.array(word_embedding).mean(axis=0)

                    #already find the word
                    if word in vocab:
                        if word not in word_embeddings_sum:
                            word_embeddings_sum[word] = []
                        word_embeddings_sum[word].append(word_embedding)
                    word = ""
                    word_embedding = []

    # calculate the average embedding for each word
    word_embeddings_sum = {word: word_embeddings_sum[word] for word in word_embeddings_sum if len(word_embeddings_sum[word]) >= 5}
    word_embeddings_result = {word: np.mean(np.array(word_embeddings_sum[word]),axis=0) for word in word_embeddings_sum}

    return word_embeddings_result


def get_word_embeddings(corpus_dir, output_dir, model_dir, tokenizer_dir, vocab_path):
    """
    Extracts and saves word embeddings for each corpus using specified models.

    :param corpus_dir: Directory containing text files for the corpus.
    :param output_dir: Directory where the computed word embeddings will be saved.
    :param model_dir: Directory containing the pre-trained model files.
    :param tokenizer_dir: Directory containing the tokenizer files.
    :param vocab_path: File containing the vocab.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))

    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    
    # load vocab
    vocab = set()
    with open(vocab_path) as f:
        for line in f:
            vocab.add(line.strip())
    print(f"vocab length: {len(vocab)}")
        
    # load tokenizer
    print("loading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)
    
    # load corpora and models
    corpus_path_lst = get_absolute_file_paths(corpus_dir)

    if is_single_model_path(model_dir):
        model_path_lst = [model_dir]
    else:
        model_path_lst = get_absolute_file_paths(model_dir)
    
    for corpus_path in corpus_path_lst:
        corpus = list(read_corpus(corpus_path))
        corpus = [chunk for doc in corpus for chunk in chunk_text(doc, tokenizer)]
        corpus_name = os.path.basename(corpus_path).split(".")[0].lower()
        for model_path in model_path_lst:
            model_name = os.path.basename(model_path).split(".")[0].lower()
            model = BertModel.from_pretrained(
                model_path, output_hidden_states=True
            ) 
            model = model.to(device)
            model.eval()
            # get word embeddings
            print(f"processing inputs from {corpus_name} using {model_name} ...")
            word_embeddings_avg = seq_to_token_embeddings(
                tokenizer=tokenizer, model=model, corpus=corpus, device=device, vocab=vocab
            )
            # save word embeddings
            model_save_path = os.path.join(
                output_dir, f"{corpus_name}_{model_name}.vec"
            )
            save_embeddings(model_save_path, word_embeddings_avg)
            print(f"model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract word embeddings using BERT.")
    parser.add_argument("-t", "--tokenizer_dir", type=str, required=True, help="Path to the pretrained tokenizer.")
    parser.add_argument("-m", "--model_dir", type=str, required=True, help="Directory containing pretrained BERT models.")
    parser.add_argument("-c", "--corpus_dir", type=str, required=True, help="Directory containing corpus files.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save the extracted embeddings.")
    parser.add_argument("-v", "--vocab_path", type=str, required=True, help="File containing vocab.")


    args = parser.parse_args()

    if not os.path.exists(args.tokenizer_dir):
        raise ValueError(f"Tokenizer directory does not exist: {args.tokenizer_dir}")

    if not os.path.exists(args.model_dir):
        raise ValueError(f"Model directory does not exist: {args.model_dir}")

    if not os.path.exists(args.corpus_dir):
        raise ValueError(f"Corpus directory does not exist: {args.corpus_dir}")
    
    if not os.path.exists(args.vocab_path):
        raise ValueError(f"Vocab file does not exist: {args.vocab_path}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    get_word_embeddings(args.corpus_dir, args.output_dir, args.model_dir, args.tokenizer_dir, args.vocab_path)
