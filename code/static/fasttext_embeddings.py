import os
from gensim.models import FastText
from tqdm import tqdm
from gensim.models import KeyedVectors
import argparse

ang_text_path = "data/AngText"
eng_text_path = "data/EngText"
eng2_text_path = "data/EngText2"
base_text_path = "data/AllText"
pretrained_dir = "pretrained_vec/"
vocab_path = "data/vocab.txt"


def read_corpus(file_path):
    with open(file_path) as f:
        for line in f:
            yield line.strip().split()

def train_fasttext(file_path, 
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
        'negative': 10, 
        'ns_exponent': 0.75, 
        'alpha': 0.05,  
        'min_alpha': 0.001,  
        'epochs': epochs,
        'sample': 1e-3, 
        'min_n': 3, 
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


def continue_train(pre_trained_path, 
                   file_path, 
                   output_dir="outputs/",
                   model_path="default.model"):
    """
    Continue training the pretrained embedding.
    """
    continue_epoch = 30
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
    model.train(corpus, total_examples=len(corpus), epochs=continue_epoch)
    merged_model_path = os.path.join(output_dir, model_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(merged_model_path)


def run_internal(vecsize=300, base_epochs=50):
    '''
    a wrapper function to run training based on all text and then continue training based on ang and eng text.
    '''
    all_model_path = f"all_fasttext_{vecsize}.model"
    output_dir = f"outputs/internal_{vecsize}/"
    all_pretrained_path = output_dir + all_model_path
    ang_model_path = f"ang_internal_{vecsize}.model"
    eng_model_path = f"eng_internal_{vecsize}.model"
    eng2_model_path = f"eng2_internal_{vecsize}.model"
    train_fasttext(file_path=base_text_path, 
                   output_dir=output_dir, 
                   model_path=all_model_path, 
                   vecsize=vecsize, 
                   epochs=base_epochs)
    continue_train(pre_trained_path=all_pretrained_path,
                   file_path=ang_text_path, 
                   output_dir=output_dir, 
                   model_path=ang_model_path)
    continue_train(pre_trained_path=all_pretrained_path,
                   file_path=eng_text_path, 
                   output_dir=output_dir, 
                   model_path=eng_model_path)
    continue_train(pre_trained_path=all_pretrained_path,
                     file_path=eng2_text_path, 
                     output_dir=output_dir, 
                     model_path=eng2_model_path)
    return output_dir + ang_model_path, output_dir + eng_model_path, output_dir + eng2_model_path
    
def run_external(vecsize=300):
    '''
    a wrapper function to run continue training based on fasttext latin model.
    '''
    pretrained_path = pretrained_dir + f"cc.la.{vecsize}.bin"
    output_dir = f"outputs/external_{vecsize}/"
    base_path = f"base_external_{vecsize}.model"
    ang_model_path = f"ang_external_{vecsize}.model"
    eng_model_path = f"eng_external_{vecsize}.model"
    eng2_model_path = f"eng2_external_{vecsize}.model"
    continue_train(pre_trained_path=pretrained_path,
                     file_path=base_text_path, output_dir=output_dir, model_path=base_path)
    continue_train(pre_trained_path=output_dir + base_path,
                   file_path=ang_text_path, output_dir=output_dir, model_path=ang_model_path)
    continue_train(pre_trained_path=output_dir + base_path,
                   file_path=eng_text_path, output_dir=output_dir, model_path=eng_model_path)
    continue_train(pre_trained_path=output_dir + base_path,
                    file_path=eng2_text_path, output_dir=output_dir, model_path=eng2_model_path)
    return output_dir + ang_model_path, output_dir + eng_model_path, output_dir + eng2_model_path

def shrink_to_vec(model_path="outputs/ang_external_300.model", vocab_path = vocab_path):
    '''
    Keep the necessary word vectors and convert the model to vec format for evaluation.
    '''
    model = FastText.load(model_path)
    selected_words = [word for line in list(read_corpus(vocab_path)) for word in line]
    new_model = KeyedVectors(vector_size=model.vector_size)
    for word in tqdm(selected_words, desc="Shrinking" + model_path):
        if word in model.wv:
            new_model.add_vector(word, model.wv.get_vector(word))
    output_path = model_path.replace(".model", ".vec")
    model.wv.save_word2vec_format(output_path, binary=False)    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate FastText embeddings.')
    parser.add_argument('--vecsize', type=int, default=300, help='Size of the vector embeddings.')
    parser.add_argument('--internal', action='store_true', help='Whether to run internal initialization training.')
    args = parser.parse_args()
    if args.internal:
        mp1, mp2, mp3 = run_internal(vecsize=args.vecsize)
    else:
        mp1, mp2, mp3 = run_external(vecsize=args.vecsize)
    shrink_to_vec(mp1)
    shrink_to_vec(mp2)
    shrink_to_vec(mp3)