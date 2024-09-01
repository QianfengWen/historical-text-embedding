import os
from gensim.models import FastText
from tqdm import tqdm
from gensim.models import KeyedVectors
import argparse

MODE = ["internal", "external", "incremental"]
pretrained_dir = "pretrained_vec/"

def read_corpus(file_path):
    with open(file_path, encoding='utf-8', errors='ignore') as f:
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

    merged_model_path = os.path.join(output_dir, model_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(merged_model_path)
    print("model saved")


def continue_train(pre_trained_path,
                   file_path, 
                   output_dir="outputs/",
                   model_path="default.model",
                   vecsize=300,
                   continue_epoch=30):
    """
    Continue training the pretrained embedding.
    """
    # continue_epoch = 30
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


def run_internal(vecsize=300, base_epochs=50, continue_epochs=50, period_files=[], base_text_path="data/AllText"):
    '''
    a wrapper function to run training based on all text and then continue training based on ang and eng text.
    '''
    all_model_path = f"all_fasttext_{vecsize}.model"
    output_dir = f"outputs/internal_{vecsize}/"
    pretrained_path = output_dir + all_model_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Training internal")
    # train the model based on all text
    train_fasttext(file_path=base_text_path, 
                       output_dir=output_dir, 
                       model_path=all_model_path, 
                       vecsize=vecsize, 
                       epochs=base_epochs)
    print("Training internal done")
    result_list = []
    # continue training based on ang and eng text
    for file in period_files:
        model_path = file.split("/")[-1].lower() + f"_internal_{vecsize}_{continue_epochs}.model"
        continue_train(pre_trained_path=pretrained_path,
                       file_path=file, 
                       output_dir=output_dir, 
                       model_path=model_path,
                       continue_epoch=continue_epochs,
                       vecsize=vecsize)
        result_list.append(output_dir + model_path)
        # update the pretrained path
        pretrained_path = output_dir + model_path
    return result_list
    
def run_external(vecsize=300, continue_epochs=50, period_files=[], base_text_path="data/AllText"):
    '''
    a wrapper function to run continue training based on fasttext latin model.
    '''
    pretrained_path = pretrained_dir + f"cc.la.{vecsize}.bin"
    output_dir = f"outputs/external_{vecsize}/"
    base_path = f"base_external_{vecsize}.model"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # continue training based on all text
    continue_train(pre_trained_path=pretrained_path,
                     file_path=base_text_path, 
                     output_dir=output_dir, 
                     model_path=base_path,
                     continue_epoch=continue_epochs,
                     vecsize=vecsize)
    
    result_list = []
    # continue training based on ang and eng text
    # reverse the order of period_files
    reversed_period_files = period_files[::-1]
    for file in reversed_period_files:
        model_path = file.split("/")[-1].lower() + f"_external_{vecsize}_{continue_epochs}.model"
        continue_train(pre_trained_path=output_dir + base_path,
                       file_path=file, 
                       output_dir=output_dir, 
                       model_path=model_path,
                       continue_epoch=continue_epochs,
                       vecsize=vecsize)
        result_list.append(output_dir + model_path)
        # update the pretrained path
        pretrained_path = output_dir + model_path
    return result_list

def run_incremental(vecsize=300, base_epochs=50, continue_epochs=50, period_files=[], base_text_path="data/AllText"):
    '''
    Run incremental training based on the previous period in a reversed order.
    '''
    print("Running incremental training")
    output_dir = f"outputs/incremental_{vecsize}/"
    # first train the model based on the last period
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_list = []
    print("Training last period")
    train_fasttext(file_path=period_files[-1], 
                   output_dir=output_dir, 
                   model_path=period_files[-1].split("/")[-1].lower() + f"_incremental_{vecsize}_{continue_epochs}.model", 
                   vecsize=vecsize, 
                   epochs=continue_epochs)
    result_list.append(output_dir + period_files[-1].split("/")[-1].lower() + f"_incremental_{vecsize}_{continue_epochs}.model")
    print("Training last period done")
    # continue training based on the previous period
    for i in range(len(period_files) - 2, -1, -1):
        print(f"Continue training period {i}")
        model_path = period_files[i].split("/")[-1].lower() + f"_incremental_{vecsize}_{continue_epochs}.model"
        continue_train(pre_trained_path=result_list[-1],
                       file_path=period_files[i], 
                       output_dir=output_dir, 
                       model_path=model_path,
                       continue_epoch=continue_epochs,
                       vecsize=vecsize)
        result_list.append(output_dir + model_path)
        print(f"Continue training period {i} done")
    return result_list

def shrink_to_vec(model_path="outputs/ang_external_300.model", vocab_path = "data/vocab.txt"):
    '''
    Keep the necessary word vectors and convert the model to vec format for evaluation.
    '''
    print("Shrinking" + model_path)
    model = FastText.load(model_path)
    selected_words = [word for line in list(read_corpus(vocab_path)) for word in line]
    new_model = KeyedVectors(vector_size=model.vector_size)
    for word in tqdm(selected_words, desc="Shrinking" + model_path):
        if word in model.wv:
            new_model.add_vector(word, model.wv.get_vector(word))
    output_path = model_path.replace(".model", ".vec")
    new_model.save_word2vec_format(output_path, binary=False)    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate FastText embeddings.')
    parser.add_argument('--vecsize', type=int, default=300, help='Size of the vector embeddings.')
    parser.add_argument('--continue_epochs', type=int, default=50, help='Number of epochs for training.')
    parser.add_argument('--mode', type=str, default='internal', choices=MODE, help='Training mode.')
    parser.add_argument('--data_folder', type=str, default='data/', help='Folder containing the data.')
    args = parser.parse_args()
    base_text_path = args.data_folder + "AllText"
    vocab_path = args.data_folder + "vocab.txt"
    print("base_text_path: ", base_text_path)
    print("vocab_path: ", vocab_path)
    text_dir = args.data_folder + "periods/"
    # for each file in period_files, store their path in a list
    period_files = [text_dir + file for file in os.listdir(text_dir)]
    mp_list = []
    if args.mode == "internal":
        # mp1, mp2, mp3 = run_internal(vecsize=args.vecsize, period_files=period_files, base_text_path=base_text_path)
        mp_list = run_internal(vecsize=args.vecsize, continue_epochs=args.continue_epochs, period_files=period_files, base_text_path=base_text_path)
    elif args.mode == "external":
        # mp1, mp2, mp3 = run_external(vecsize=args.vecsize, period_files=period_files, base_text_path=base_text_path)
        mp_list = run_external(vecsize=args.vecsize, continue_epochs=args.continue_epochs, period_files=period_files, base_text_path=base_text_path)
    elif args.mode == "incremental":
        # mp1, mp2, mp3 = run_incremental(vecsize=args.vecsize, period_files=period_files, base_text_path=base_text_path)
        mp_list = run_incremental(vecsize=args.vecsize, continue_epochs=args.continue_epochs, period_files=period_files, base_text_path=base_text_path)
    for mp in mp_list:
        shrink_to_vec(mp, vocab_path)