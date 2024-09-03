# Diachronic Embeddings for Medieval Latin Charters

This project applies various embedding methods to Medieval Latin charters from the Norman Conquest period to analyze social and cultural changes during this era through a computational linguistic lens.

Paper available at [[Paper]](lexical_semantic_change_for_medieval_latin.pdf)

## Setup

### Installing Required Packages

To install the required packages for this project, run the following command:

```bash
pip install -r requirements.txt
```

### Accessing Data
The Medieval Latin charters used in this project are available in the `data/` folder. Please refer to the [DEEDS website](https://deeds.library.utoronto.ca/content/about-deeds) for more details.

For semantic change labels, please refer to `data/labels/change.txt` for the list of words in the "changed" group and `data/labels/unchange.txt` for the list of words in the "unchanged" group.


## Usage

### Models

#### FastText Embeddings

The default setup uses external initialization with pre-trained FastText vectors (available at [FastText Crawl Vectors](https://fasttext.cc/docs/en/crawl-vectors.html)). Use `--internal` for internal initialization.

```bash
python code/static/fasttext_embeddings.py --vecsize <hidden_size> [--internal]
```

#### BERT Embeddings

- **Train BERT models:**
The default setup uses fine-tuning method using a specified model (`-m`) and a pre-trained tokenizer (`-t`). Use the `--pretrain` flag to train the model from scratch and `--train_tokenizer` to train a new tokenizer.

  ```bash
  python code/bert/runner.py -i path/to/corpus.txt -o path/to/output [-m path/to/your/base/model] -t path/to/your/tokenizer [--pretrain] [--train_tokenizer]
  ```

- **Extract word embeddings from BERT models:**
By default, embeddings are aggregated using the `mean` method. To use `max` or `min` aggregation, specify it with the `-a` option.

  ```bash
  python code/bert/extract_embeddings.py -t /path/to/pretrained/tokenizer -m /path/to/pretrained/models -c /path/to/corpus/files -o /path/to/save/embeddings [-a aggregation/method]

  ```