from code.utils import *

from statsmodels.stats.proportion import proportions_ztest
from collections import Counter

def filter_date(corpus_path, time_path, new_corpus_path, start_date, end_date):
    """Filters a text corpus based on date range and writes the filtered corpus to a new file."""
    dates = read_time(time_path)
    corpus = read_corpus_no_split(corpus_path)

    with open(new_corpus_path, "w") as f:
        for doc, date in zip(corpus, dates):
            if start_date <= date <= end_date:
                f.write(f"{doc}\n")


def build_counter(ang_path, eng_path):
    """Builds word counters for two corpora."""
    ang_counter = Counter()
    eng_counter = Counter()
    
    # read and count words in ang_path
    for line in read_corpus_no_split(ang_path):
        words = line.split()  
        ang_counter.update(words)
    
    # read and count words in eng_path
    for line in read_corpus_no_split(eng_path):
        words = line.split()  
        eng_counter.update(words)
    
    return ang_counter, eng_counter

def compute_rel_freq(counter, total_count):
    """Computes the relative frequencies for words."""
    return {word: count / total_count for word, count in counter.items()}


def create_word_sets(ang_path, eng_path, words_path):
    """Selects common words with a frequency of at least 50 in both counters."""
    ang_counter, eng_counter = build_counter(ang_path, eng_path)

    ang_total = sum(ang_counter.values())
    eng_total = sum(eng_counter.values())

    ang_rel_freq = compute_rel_freq(ang_counter, ang_total)
    eng_rel_freq = compute_rel_freq(eng_counter, eng_total)

    ang_set = {word for word, freq in ang_rel_freq.items() if freq >= 0.0002}
    eng_set = {word for word, freq in eng_rel_freq.items() if freq >= 0.0002}

    # find the intersection of words in both sets
    word_set = ang_set.intersection(eng_set)

    with open(words_path, "w") as f:
        for word in word_set:
            f.write(f"{word}\n")

    with open(f"stats_{words_path}", "w") as f:
        for word in word_set:
            ang_count = ang_counter.get(word, 0)
            eng_count = eng_counter.get(word, 0)

            counts = [ang_count, eng_count]
            nobs = [ang_total, eng_total]

            p1 = ang_count / ang_total
            p2 = eng_count / eng_total

            # perform two-proportion z-test 
            _, p_value = proportions_ztest(counts, nobs)
            f.write(f"{word}, {p1:.4f}, {p2:.4f}, {p1 - p2:.4f}, {p_value:.4f}\n")

    
if __name__ == "__main__":
    ang_path = "data/AngText"
    eng_path = "data/EngText"
    vocab_path = "data/vocab.txt"

    ang_time_path = "data/AngOrdDate"
    eng_time_path = "data/EngOrdDate"

    filtered_ang_path = "data/corpus/AngText"
    filtered_eng_path = "data/corpus/EngText"

    # filter_date(ang_path, ang_time_path, filtered_ang_path, 800, 1066)
    # filter_date(eng_path, eng_time_path, filtered_eng_path, 1066, 1220)

    create_word_sets(filtered_ang_path, filtered_eng_path, vocab_path)

    