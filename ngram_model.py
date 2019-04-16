#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)


def add_to_dict(key,dict):
    if key in dict.keys():
        dict[key] += 1
        return 0
    else:
        dict[key] = 1
        return 1

def size_of_corpus(unigrams):
    count=0
    for word in unigrams:
        count += unigrams[word]
    return count

def generate_ngrams(s, n):
    #return zip(*[s[i:] for i in range(n)])
    i=0
    while i<=len(s)-n:
        # if n==1 and (i==0 or i==1):
        #     continue
        # if n==2 and i==0:
        #     continue
        yield tuple([s[i+j] for j in range(n)])
        i+=1

def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    token_count = 0

    for sentence in dataset[:1000]:
        a = ['*','*']
        a.extend(list(sentence))
        sentence = a
        # # unigrams
        # for word in sentence:
        #     token_count += add_to_dict(word,unigram_counts)
        # # bigrams
        # sentence_bigrams = generate_ngrams(sentence,2)
        # for bigram in sentence_bigrams:
        #     token_count += add_to_dict(bigram,bigram_counts)
        # # trigrams
        # sentence_trigrams = generate_ngrams(sentence,3)
        # for trigram in sentence_trigrams:
        #     token_count += add_to_dict(trigram,trigram_counts)

        for j in range(2,len(sentence)):
            add_to_dict((sentence[j-2],sentence[j-1],sentence[j]),trigram_counts)
            add_to_dict((sentence[j-1], sentence[j]), bigram_counts)
            add_to_dict((sentence[j]), unigram_counts)

    return trigram_counts, bigram_counts, unigram_counts, token_count

def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    M = size_of_corpus(unigram_counts)
    #perplexity = 0
    mult = 1
    for sentence in eval_dataset[:1000]:
        a = ['*', '*']
        a.extend(list(sentence))
        sentence = a
        for j in range(2,len(sentence)):
            b = 0
            if (sentence[j-2],sentence[j-1],sentence[j]) in trigram_counts:
                q_tri = float(trigram_counts[(sentence[j-2],sentence[j-1],sentence[j])]) / bigram_counts[(sentence[j-2],sentence[j-1])]
                q_bi = float(bigram_counts[(sentence[j-1],sentence[j])]) / unigram_counts[(sentence[j-1])]
                q_uni = float(unigram_counts[(sentence[j])]) / M
                mult *= (lambda1*q_tri + lambda2*q_bi + (1-lambda1-lambda2)*q_uni)
    perplexity = 2**(np.log2(mult) / M)
    return perplexity

def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    #Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)
    ### YOUR CODE HERE
    ### END YOUR CODE

if __name__ == "__main__":
    test_ngram()