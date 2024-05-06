import os, random, argparse
from code.utils import *

# def combine_corpus(ang_path, eng_path):
#     """
#     Combines corpora into a single list of documents.

#     :param ang_path: Path to the 'ang' corpus file.
#     :param eng_path: Path to the 'eng' corpus file.
#     """
#     ang = list(read_corpus_no_split(ang_path))
#     eng = list(read_corpus_no_split(eng_path))

#     combined = ang + eng
#     ratio = len(ang) / len(combined)

#     return combined, ratio


# def split_corpus(combined_corpus, seed, split_ratio):
#     """
#     Splits the combined corpus into two subsets, maintaining the specified label ratio.

#     :param combined_corpus: List of texts representing the combined corpus.
#     :param seed: Random seed for reproducibility.
#     :param split_ratio: Ratio of the 'ang' data in the first subset.
#     """
#     random.seed(seed)
#     random.shuffle(combined_corpus)

#     # calculate split indices
#     split_idx = int(len(combined_corpus) * split_ratio)

#     # create the subsets
#     subset1 = combined_corpus[:split_idx]
#     subset2 = combined_corpus[split_idx:]
#     random.shuffle(subset1)
#     random.shuffle(subset2)

#     return subset1, subset2

def generate_data(ang_path, eng_path, corpus_path, repeat=50):
    """
    Generates multiple random splits of the combined corpus and writes them to files.

    :param ang_path: Path to the 'ang' corpus file.
    :param eng_path: Path to the 'eng' corpus file.
    :param corpus_path: Directory to save the generated data files.
    :param repeat: Number of times to repeat the data generation.
    """
    ang = list(read_corpus_no_split(ang_path))
    eng = list(read_corpus_no_split(eng_path))
    combined = ang + eng

    # introduce noise to dataset
    def modify_corpus(corpus):
        return [
            random.choice(combined) if random.random() < 0.5 else doc
            for doc in corpus
        ]
    
    # repeat data generation
    for i in range(repeat):
        random.seed(i)
        ang_new = modify_corpus(ang)
        eng_new = modify_corpus(eng)

        ang_filename = os.path.join(corpus_path, f"AngRandom{i+1}.txt")
        eng_filename = os.path.join(corpus_path, f"EngRandom{i+1}.txt")

        with open(ang_filename, "w") as f_a:
            f_a.write("\n".join(ang_new))
        
        with open(eng_filename, "w") as f_e:
            f_e.write("\n".join(eng_new))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random datasets from text corpora.')
    parser.add_argument('-a', '--ang_path', type=str, help='Path to the ang corpus file.')
    parser.add_argument('-e', '--eng_path', type=str, help='Path to the eng corpus file.')
    parser.add_argument('-c', '--corpus_path', type=str, help='Path to save the generated data files.')
    parser.add_argument('--repeat', type=int, default=50, help='Number of repetitions for data generation.')
    
    args = parser.parse_args()

    if not os.path.isfile(args.ang_path):
        raise ValueError(f"File '{args.ang_path}' does not exist.")

    if not os.path.isfile(args.eng_path):
        raise ValueError(f"File '{args.eng_path}' does not exist.")
    
    os.makedirs(args.corpus_path, exist_ok=True)

    if args.repeat <= 0:
        raise ValueError("The 'repeat' parameter must be a positive integer.")

    args = parser.parse_args()

    generate_data(ang_path=args.ang_path, eng_path=args.eng_path, corpus_path=args.corpus_path, repeat=args.repeat)

        

