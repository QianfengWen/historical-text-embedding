# historical-text-embedding

Standardization usage



For standardizing Ang
python standardization.py -i data/AngOrdtext -d experiments/result/AngDict -o experiments/result/AngStandText
For standardizing Eng
python standardization.py -i data/EngOrdtext -d experiments/result/EngDict -o experiments/result/EngStandText

For static model
python code/skipgram_embeddings.py --vecsize 100 --epochs 30 --internal
python code/skipgram_embeddings.py --vecsize 100 --epochs 50 --internal
python code/skipgram_embeddings.py --vecsize 300 --epochs 30 --internal
python code/skipgram_embeddings.py --vecsize 300 --epochs 50 --internal
python code/skipgram_embeddings.py --vecsize 100 --epochs 30 
python code/skipgram_embeddings.py --vecsize 300 --epochs 30
