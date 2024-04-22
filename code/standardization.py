from Levenshtein import ratio
from tqdm import tqdm
import random
from collections import Counter
import argparse

def get_unique_words(filepath):    
    '''
    Read a file and return a set of unique words
    '''
    with open(filepath, "r") as f:
        lines = f.readlines()
        return {word for line in lines for word in line.split()}
    
def standardize_letters(sentence):
    '''
    Standardize letters in a sentence
    '''
    # replace other letters
    word_map2 = {'æ': 'e', 'ae': 'e', 'ð': 'th', 'þ': 'th'}
    new_sentence = []
    for word in sentence:
        new_word = word
        for letter in new_word: 
            if letter in word_map2:
                new_word = new_word.replace(letter, word_map2[letter])
        new_sentence.append(new_word)
    return new_sentence
    
def get_freq(filepath):
    '''
    Get frequency of words in a file
    '''
    with open(filepath, "r") as f:
        lines = f.readlines()
        return Counter([word for line in lines for word in standardize_letters(line.split())])

def find_similar_words(scarce, not_scarce):
    '''
    Return a dictionary of format {scarce_word: most_similar_and_not_scarce_word}
    '''
    normal = {}
    for word in tqdm(scarce):
        max_word = max(not_scarce, key=lambda x: ratio(word, x))
        max_sim = ratio(word, max_word)
        if max_sim > 0.85:
            normal[word] = max_word
    return normal

def original_to_dict(input_file, dict_file):
    '''
    Standardize the text in a file
    '''
    freq = get_freq(input_file)
    scarce = {word for word in freq if freq[word] < 5}
    not_scarce = {word for word in freq if freq[word] >= 5}
    dict = find_similar_words(scarce, not_scarce)
    with open(dict_file, "w") as file:
        for key, value in dict.items():
            file.write(f"{key}:{value}\n")


def dict_to_standard(input_file, dict_file, output_file):
    with open(dict_file, "r") as f:
        lines = f.readlines()
        dict = {}
        for line in lines:
            key, value = line.split(":")
            dict[key] = value.strip()
    with open(input_file, "r") as f:
        lines = f.readlines()
        with open(output_file, "w") as out:
            for line in lines:
                words = standardize_letters(line.split())
                new_line = []
                for word in words:
                    if word in dict:
                        new_line.append(dict[word])
                    else:
                        new_line.append(word)
                out.write(" ".join(new_line) + "\n")
    
def combine_files(file1, file2, output_file):
    '''
    combine two files into one
    '''
    with open(file1, "r") as f1, \
            open(file2, "r") as f2, \
            open(output_file, "w") as out:
        
        out.write(f1.read() + "\n" + f2.read())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standardize the text in a file.')
    parser.add_argument('-i', type=str, help='Path to the input file.')
    parser.add_argument('-d', type=str, help='Path to the dictionary file.')
    parser.add_argument('-o', type=str, help='Path to the second output file.')

    args = parser.parse_args()
    original_to_dict(args.i, args.d) 
    dict_to_standard(args.i, args.d, args.o)
    # stand('data/AngOrdtext','experiments/result/AngDict', 'experiments/result/AngStandText')
    # stand('data/EngOrdtext','experiments/result/EngDict' ,'experiments/result/EngStandText')
    