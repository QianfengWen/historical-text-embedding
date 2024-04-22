import os
from gensim.models import FastText
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors
import argparse

def read_corpus(file_path):
    with open(file_path) as f:
        for line in f:
            yield line.strip().split()

def train_fasttext(file_path="inputs/AllStandText", 
                   output_dir="outputs/", 
                   model_path="fasttext.model",
                   vecsize=100,
                   epochs=30):
    """
    Train the  `fastText` subword model from scratch.
    """
    params = {
        'vector_size': vecsize, 
        'window': 5,
        'min_count': 5,
        'workers': 8, 
        'sg': 0, 
        'hs': 0, 
        'negative': 5, 
        'ns_exponent': 0.75, 
        'alpha': 0.025,  
        'min_alpha': 0.0001,  
        'epochs': epochs,
        'sample': 1e-3, 
        'min_n': 5, 
        'max_n': 5, 
        'bucket': 2000000, 
    }
    corpus = list(tqdm(read_corpus(file_path), desc="Reading Corpus"))
    model = FastText(corpus, **params)

    print("model trained")
    merged_model_path = os.path.join(output_dir, model_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(merged_model_path)
    print("model saved")


def continue_train(pre_trained_path="outputs/all_fasttext.model", 
                   file_path="inputs/AngOrdtext", 
                   output_dir="outputs/",
                   model_path="default.model"):
    """
    Continue training the pretrained embedding.
    """
    print("continue_train")
    if pre_trained_path.endswith('.bin'):
        # pretrained fasttext model is in binary format
        model = FastText.load_fasttext_format(pre_trained_path)
        corpus = list(tqdm(read_corpus(file_path), desc="Reading Corpus"))
    else: # pre_trained_path.endswith('.model')
        model = FastText.load(pre_trained_path)
        corpus = list(tqdm(read_corpus(file_path), desc="Reading Corpus"))
    print("pretrained model loaded")
    model.build_vocab(corpus, update=True)
    model.train(corpus, total_examples=len(corpus), epochs=30)
    merged_model_path = os.path.join(output_dir, model_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(merged_model_path)


def run_internal(vecsize=100, epochs=30):
    '''
    a wrapper function to run training based on all text and then continue training based on ang and eng text.
    '''
    all_model_path = f"all_fasttext_{vecsize}_e{epochs}.model"
    all_pretrained_path = f"outputs/internal_{vecsize}_e{epochs}/all_fasttext_{vecsize}_e{epochs}.model"
    ang_model_path = f"ang_internal_{vecsize}_e{epochs}.model"
    eng_model_path = f"eng_internal_{vecsize}_e{epochs}.model"
    train_fasttext(file_path="inputs/BaseText", 
                   output_dir=f"outputs/internal_{vecsize}_e{epochs}/", 
                   model_path=all_model_path, 
                   vecsize=vecsize, 
                   epochs=epochs)
    continue_train(pre_trained_path=all_pretrained_path,
                   file_path="inputs/AngStandText", 
                   output_dir=f"outputs/internal_{vecsize}_e{epochs}/", 
                   model_path=ang_model_path)
    continue_train(pre_trained_path=all_pretrained_path,
                   file_path="inputs/EngStandText", 
                   output_dir=f"outputs/internal_{vecsize}_e{epochs}/", 
                   model_path=eng_model_path)
    return ang_model_path, eng_model_path
    
def run_external(vecsize=100):
    '''
    a wrapper function to run continue training based on fasttext latin model.
    '''
    continue_train(pre_trained_path=f"pretrained_vec/cc.la.{vecsize}.bin",
                   file_path="inputs/AngStandText", output_dir="outputs/", model_path=f"ang_external_{vecsize}.model")
    continue_train(pre_trained_path=f"pretrained_vec/cc.la.{vecsize}.bin",
                   file_path="inputs/EngStandText", output_dir="outputs/", model_path=f"eng_external_{vecsize}.model")
    return f"outputs/ang_external_{vecsize}.model", f"outputs/eng_external_{vecsize}.model"

def convert_to_vec(model_path="outputs/ang_external_100.model"):
    '''
    Convert the model to vec format for evaluation.
    '''
    model = FastText.load(model_path)
    output_path = model_path.replace(".model", ".vec")
    model.wv.save_word2vec_format(output_path, binary=False)    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate FastText embeddings.')
    parser.add_argument('--vecsize', type=int, default=100, help='Size of the vector embeddings.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training.')
    parser.add_argument('--internal', default=False, help='Whether to run internal initialization training.')
    args = parser.parse_args()
    if args.internal:
        mp1, mp2 = run_internal(vecsize=args.vecsize, epochs=args.epochs)
    else:
        mp1, mp2 = run_external(vecsize=args.vecsize)
    convert_to_vec(mp1)
    convert_to_vec(mp2)
