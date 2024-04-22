# historical-text-embedding

Standardization usage

For standardizing Anglo-Saxon period corpora
```bash
python code/standardization.py -i data/AngOrdtext -d result/AngDict -o result/AngStandText
```

For static model, use `--internal` for internal embedding.
```bash
python code/skipgram_embeddings.py --vecsize 300 --epochs 50 --internal
```
