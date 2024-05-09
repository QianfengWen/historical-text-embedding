# Diachronic Embeddings for Medieval Latin Charters

This project applies various embedding methods to Medieval Latin charters from the Norman Conquest period to analyze social and cultural changes during this era through a computational linguistic lens.

Paper available at [[Paper]](diachronic_word_embeddings_for_medieval_Latin.pdf)

## Setup

### Installing Required Packages

To install the required packages for this project, run the following command:

```bash
pip install -r requirements.txt
```

### Accessing Data
The data used in this project is not publicly available. If you need access to the relevant data, please contact the author at [yifanliu.liu@mail.utoronto.ca](mailto:yifanliu.liu@mail.utoronto.ca).

### Organizing Data Files

Please ensure that your data folders are organized as follows:

- `data/`
  - `corpus/`
  - `metadata/`
    - `ang/` (metadata for Anglo-Saxon charters)
    - `eng/` (metadata for Norman charters)


## Usage

The project includes the training and adaptation of models, evaluation of model performance, and a semantic change detection application.

### Standardization

For standardizing Anglo-Saxon period corpora, run the following comment:

```bash
python code/data_preprocessing/standardization.py -i path/to/your/data -d path/to/your/standerdization/dict -o path/to/your/result
```

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

### Evaluate Models

#### Extrinsic Evaluations through Text Classification

The default setup processes all metadata in the corresponding folder. To evaluate using a specific label, include the `-l` option to specify the file name.

  ```bash
  python code/evaluation/text_classification.py -m path/to/model -c path/to/corpus.txt -ld path/to/label_dir -o path/to/output [-l specific_label_file_name.txt]
  ```
