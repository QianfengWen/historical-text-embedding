# historical-text-embedding

Standardization usage

For standardizing Ang
```bash
python code/standardization.py -i data/AngOrdtext -d experiments/result/AngDict -o experiments/result/AngStandText
```
For standardizing Eng
```bash
python code/standardization.py -i data/EngOrdtext -d experiments/result/EngDict -o experiments/result/EngStandText
```

For static model
```bash
python code/skipgram_embeddings.py --vecsize 300 --epochs 50 --internal
```
