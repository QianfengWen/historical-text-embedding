import os, random, argparse
from code.utils import *

def read_corpus(file_path):
    """
    Yield sentences from a corpus file.
    
    :param file_path: Path to the corpus file.
    :return: Yields lists of words in each line.
    """
    with open(file_path) as f:
        for line in f:
            yield line.strip()

def combine_corpus(ang_path, eng_path):
    """
    Combines 'ang' and 'eng' corpora into a single list of labeled documents.

    :param ang_path: Path to the 'ang' corpus file.
    :param eng_path: Path to the 'eng' corpus file.
    """
    ang = list(read_corpus(ang_path))
    eng = list(read_corpus(eng_path))

    ang = [(doc, "ang") for doc in ang]
    eng = [(doc, "eng") for doc in eng]
    combined = ang + eng

    return combined


def split_corpus(combined_corpus, seed, split_ratio):
    """
    Splits the combined corpus into two subsets, maintaining the specified label ratio.

    :param combined_corpus: List of tuples (document, label) representing the combined corpus.
    :param seed: Random seed for reproducibility.
    :param split_ratio: Ratio of the 'ang' data in the first subset.
    """
    random.seed(seed)
    random.shuffle(combined_corpus)

    # separate the data by label
    ang_data = [item[0] for item in combined_corpus if item[1] == "ang"]
    eng_data = [item[0] for item in combined_corpus if item[1] == "eng"]

    # calculate split indices
    ang_split_idx = int(len(ang_data) * split_ratio)
    eng_split_idx = int(len(eng_data) * split_ratio)

    # create the subsets
    subset1 = ang_data[:ang_split_idx] + eng_data[:eng_split_idx]
    subset2 = ang_data[ang_split_idx:] + eng_data[eng_split_idx:]
    random.shuffle(subset1)
    random.shuffle(subset2)

    return subset1, subset2


def generate_data(ang_path, eng_path, corpus_path, repeat=50):
    """
    Generates multiple random splits of the combined corpus and writes them to files.

    :param ang_path: Path to the 'ang' corpus file.
    :param eng_path: Path to the 'eng' corpus file.
    :param corpus_path: Path to save the generated data files.
    :param repeat: Number of times to repeat the data generation.
    """
    combined = combine_corpus(ang_path=ang_path, eng_path=eng_path)
    
    # calculate the ratio of 'ang' in the combined data
    num_ang = sum(1 for _, label in combined if label == "ang")
    split_ratio = num_ang / len(combined)
    
    # repeat data generation
    for i in range(repeat):
        ang_filename = os.path.join(corpus_path, f"AngRandom{i+1}.txt")
        eng_filename = os.path.join(corpus_path, f"EngRandom{i+1}.txt")

        ang_new, eng_new = split_corpus(combined_corpus=combined, split_ratio=split_ratio, seed=i)

        with open(ang_filename, "w") as f_a:
            for idx, doc in enumerate(ang_new):
                if idx < len(ang_new) - 1:
                    f_a.write(f"{doc}\n")
                else:
                    f_a.write(f"{doc}")
        
        with open(eng_filename, "w") as f_e:
            for idx, doc in enumerate(eng_new):
                if idx < len(eng_new) - 1:
                    f_e.write(f"{doc}\n")
                else:
                    f_e.write(f"{doc}")


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

        

