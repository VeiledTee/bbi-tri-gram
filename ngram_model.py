import glob
import string
from typing import Tuple, Dict, List, IO

import numpy as np


def read_blogs(blog_dir: str) -> str:
    """
    Create background corpus by reading blog posts from a given directory
    :param blog_dir: Directory to where blog posts are stored
    :return: A long string
    """
    final_string: str = ""  # string to build

    for infile in sorted(glob.glob(blog_dir + "*.xml")):  # loop through files alphabetically
        cur_file: IO = open(infile, errors='ignore')  # open file
        lines: List[str] = cur_file.readlines()  # get all lines for current file
        in_post: bool = False  # bool flag
        # loop through lines
        for line in lines:
            if line == '</post>\n':  # check if in post tag
                in_post = False
            if in_post:
                final_string += " ".join(line.translate(
                    str.maketrans('', '', string.punctuation)
                ).lower().replace('\n', ' ').lstrip().split()  # remove all punctuation and extra spaces
                                         )
            if line == '<post>\n':  # post tag check
                in_post = True

    return final_string


def count_words(text: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Counts all n-grams (up to trigram) in a corpus, and adds one to each count for LaPlace smoothing
    :param text: The entire corpus
    :return: 3 dictionaries containing the counts of each unigram, bigram, and trigram, with LaPlace smoothing accounted for
    """
    text: List[str] = text.lower().split()
    # establish dicts
    unigram: Dict[str, float] = {'UNK': 0}
    bigram: Dict[str, float] = {'UNK': 0}
    trigram: Dict[str, float] = {'UNK': 0}

    # count unigram
    for i in range(len(text)):
        if text[i] not in unigram:
            unigram[str(text[i])] = 1
        else:
            unigram[text[i]] += 1
    # count bigram
    for i in range(len(text) - 1):
        if str(text[i]) + " " + str(text[i + 1]) not in bigram:
            bigram[str(text[i]) + " " + str(text[i + 1])] = 1
        else:
            bigram[str(text[i]) + " " + str(text[i + 1])] += 1
    # count trigram
    for i in range(len(text) - 2):
        if str(text[i]) + " " + str(text[i + 1]) + " " + str(text[i + 2]) not in trigram:
            trigram[str(text[i]) + " " + str(text[i + 1]) + " " + str(text[i + 2])] = 1
        else:
            trigram[str(text[i]) + " " + str(text[i + 1]) + " " + str(text[i + 2])] += 1

    return unigram, bigram, trigram


def normalize_bi(text: str, full_text: str, unigrams: dict, bigrams: dict) -> float:
    """
    Takes a bigram and returns probability based on LaPlace smoothing and unigram count of the first word in the bigram
    :param text: The bigram to calculate the probability of
    :param full_text: The entire corpus
    :param unigrams: A dictionary containing all unigrams and their counts
    :param bigrams: A dictionary containing all bigrams and their counts
    :return: The probability of selecting the third word in the trigram
    """
    full_len: int = len(set(full_text.split()))
    return np.log(bigrams[text] + (full_len/(unigrams[text.split()[0]] + full_len)))  # log of (c_i + 1)(N/(N+V))


def normalize_tri(text: str, full_text: str, bigrams: dict, trigrams: dict) -> float:
    """
    Takes a trigram and returns probability based on LaPlace smoothing and bigram count of the first two words in the trigram
    :param text: The trigram to calculate the probability of
    :param full_text: The entire corpus
    :param bigrams: A dictionary containing all bigrams and their counts
    :param trigrams: A dictionary containing all trigrams and their counts
    :return: The probability of selecting the third word in the trigram
    """
    full_len: int = len(set(full_text.split()))
    return np.log(trigrams[text] / (bigrams[text.split()[0] + " " + text.split()[1]] + full_len))  # log of (c_i + 1)(N/(N+V))


def predict(text: str, unigrams: Dict[str, float], bigrams: Dict[str, float], trigrams: Dict[str, float]) -> None:
    """
    Given text and counts of n-grams, return the birgram and trigram predictions for the most likely subsequent word
    :param text: Text to predict the next word of
    :param unigrams: A dictionary containing all unigrams and their counts
    :param bigrams: A dictionary containing all bigrams and their counts
    :param trigrams: A dictionary containing all trigrams and their counts
    """
    split_text: List[str] = text.split()

    bi = split_text[-1] if split_text[-1] in list(unigrams.keys()) else "UNK"
    tri = split_text[-2] + " " + split_text[-1] if split_text[-2] + " " + split_text[-1] in list(bigrams.keys()) else "UNK"

    bi_probs = []
    bi_words = []
    tri_probs = []
    tri_words = []

    """
    How to deal with UNK?
    - bigram: if new, convert to UNK
    - trigram: print UNK if any word is UNK in bigram
    """

    for key, value in bigrams.items():
        split_k = key.split()
        if split_k[0] == bi:
            bi_probs.append(np.log((value + 1) / (unigrams[bi] + len(list(unigrams.keys())))))
            bi_words.append(split_k[1])

    for key, value in trigrams.items():
        split_k = key.split()
        if key == "UNK":
            tri_probs.append(np.log((value + 1) / (bigrams["UNK"] + len(list(unigrams.keys())))))
            tri_words.append(split_k[0])
        elif split_k[0] + " " + split_k[1] == tri:
            tri_probs.append(np.log((value + 1) / (bigrams[tri] + len(list(unigrams.keys())))))
            tri_words.append(split_k[2])

    bi_max = np.argmax(bi_probs)
    tri_max = np.argmax(tri_probs)

    print(f"Bigram  Token: {bi_words[bi_max].ljust(9)[:9]} | {bi_probs[bi_max]}")
    print(f"Trigram Token: {tri_words[tri_max].ljust(9)[:9]} | {tri_probs[tri_max]}")


def main():
    # blogs_dir_path = sys.argv[1]  # Gets the location of the blog directory on the command line
    blogs_dir_path = 'Blogs/'
    blog_text = read_blogs(blogs_dir_path)
    unigrams, bigrams, trigrams = count_words(blog_text)
    sample_texts = ["cat chased the", "salt and", "jump over", "in the", "I can't believe", "oh my", "how did he", "this water is"]
    for sample in sample_texts:
        print(sample)
        predict(sample, unigrams, bigrams, trigrams)


if __name__ == '__main__':
    main()
