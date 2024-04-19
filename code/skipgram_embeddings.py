import os
from gensim.models import FastText
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors

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


def run_adapt(vecsize=100, epochs=30):
    '''
    a wrapper function to run training based on all text and then continue training based on ang and eng text.
    '''
    all_model_path = f"all_fasttext_{vecsize}_e{epochs}.model"
    all_pretrained_path = f"outputs/adapt_{vecsize}_e{epochs}/all_fasttext_{vecsize}_e{epochs}.model"
    ang_model_path = f"ang_adapt_{vecsize}_e{epochs}.model"
    eng_model_path = f"eng_adapt_{vecsize}_e{epochs}.model"
    train_fasttext(file_path="inputs/BaseText", 
                   output_dir=f"outputs/adapt_{vecsize}_e{epochs}/", 
                   model_path=all_model_path, 
                   vecsize=vecsize, 
                   epochs=epochs)
    continue_train(pre_trained_path=all_pretrained_path,
                   file_path="inputs/AngStandText", 
                   output_dir=f"outputs/adapt_{vecsize}_e{epochs}/", 
                   model_path=ang_model_path)
    continue_train(pre_trained_path=all_pretrained_path,
                   file_path="inputs/EngStandText", 
                   output_dir=f"outputs/adapt_{vecsize}_e{epochs}/", 
                   model_path=eng_model_path)
    
def run_157(vecsize=100):
    '''
    a wrapper function to run continue training based on fasttext latin model.
    '''
    continue_train(pre_trained_path=f"pretrained_vec/cc.la.{vecsize}.bin",
                   file_path="inputs/AngStandText", output_dir="outputs/", model_path=f"ang_157_{vecsize}.model")
    continue_train(pre_trained_path=f"pretrained_vec/cc.la.{vecsize}.bin",
                   file_path="inputs/EngStandText", output_dir="outputs/", model_path=f"eng_157_{vecsize}.model")

def convert_to_vec(model_path="outputs/ang_157_100.model"):
    '''
    Convert the model to vec format for evaluation.
    '''
    model = FastText.load(model_path)
    output_path = model_path.replace(".model", ".vec")
    model.wv.save_word2vec_format(output_path, binary=False)

def eval(model_0_path="outputs/eng_adapt_100.vec", 
         model_1_path="outputs/ang_adapt_100.vec", 
         word="domino"):
    '''
    A simple evaluation function to check if the word is in the model and if the embeddings are the same.
    '''
    model_0 = KeyedVectors.load_word2vec_format(model_0_path, binary=False)
    model_1 = KeyedVectors.load_word2vec_format(model_1_path, binary=False)
    print("Model 0 vocab size:", len(model_0.index_to_key))
    print("Model 1 vocab size:", len(model_1.index_to_key))
    if word in model_0.index_to_key:
        print(word + " in model 0")
        print(model_0.key_to_index[word])
    if word in model_1.index_to_key:
        print(word + " in model 1")
        print(model_1.key_to_index[word])
    if word in model_0.index_to_key and word in model_1.index_to_key:
        print(np.allclose(model_0[word], model_1[word], atol=1e-5))

    
    
if __name__ == "__main__":
    #convert_to_vec(model_path="outputs/eng_adapt_100.model")
    #convert_to_vec(model_path="outputs/ang_adapt_100.model")
    #convert_to_vec(model_path="outputs/eng_adapt_300.model")
    #convert_to_vec(model_path="outputs/ang_adapt_300.model")
    #convert_to_vec(model_path="outputs/eng_157_100.model")
    #convert_to_vec(model_path="outputs/ang_157_100.model")
    #convert_to_vec(model_path="outputs/eng_157_300.model")
    #convert_to_vec(model_path="outputs/ang_157_300.model")
    #convert_to_vec(model_path="outputs/adapt_100_e50/ang_adapt_100_e50.model")
    #convert_to_vec(model_path="outputs/adapt_100_e50/eng_adapt_100_e50.model")
    run_adapt(vecsize=300, epochs=30)
    #convert_to_vec(model_path="outputs/adapt_300_e50/ang_adapt_300_e50.model")
    #convert_to_vec(model_path="outputs/adapt_300_e50/eng_adapt_300_e50.model")