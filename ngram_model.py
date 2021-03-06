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
    if key in dict:
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

    for sentence in dataset:
        for j in range(2,len(sentence)):
            add_to_dict((sentence[j-2],sentence[j-1],sentence[j]),trigram_counts)
            add_to_dict((sentence[j-1], sentence[j]), bigram_counts)
            add_to_dict((sentence[j]), unigram_counts)
            token_count += 1
        add_to_dict((sentence[0], sentence[1]), bigram_counts)
        add_to_dict((sentence[0]), unigram_counts)
        add_to_dict((sentence[1]), unigram_counts)
        token_count += 2

    return trigram_counts, bigram_counts, unigram_counts, token_count

def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    # Total words in eval_dataset
    M=0
    # P is the probability that our language model gives to a sentences
    log_p = 0
    for sentence in eval_dataset:
        for j in range(2, len(sentence)):
            # Calculate q_ML of the trigram
            if (sentence[j-2],sentence[j-1],sentence[j]) in trigram_counts:
                q_tri = float(trigram_counts[(sentence[j-2],sentence[j-1],sentence[j])]) / bigram_counts[(sentence[j-2],sentence[j-1])]
            else:
                q_tri = 0

            # Calculate q_ML of the bigram
            if (sentence[j-1],sentence[j]) in bigram_counts:
                q_bi = float(bigram_counts[(sentence[j-1],sentence[j])]) / unigram_counts[(sentence[j-1])]
            else:
                q_bi = 0
            # Calculate q_ML of the unigram

            if sentence[j] in unigram_counts:
                q_uni = float(unigram_counts[sentence[j]]) / train_token_count
            else:
                q_uni = 0

            # Calculate  the linear interpolation
            q = (lambda1*q_tri + lambda2*q_bi + float((1.0-lambda1-lambda2))*q_uni)
            if q <= 0:
                continue
            M+=1
            log_p -= np.log2(q)

    perplexity = 2**(log_p / float(M))
    return perplexity

def grid_search():
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    lambdas1 = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1}
    lambdas2 = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1}
    min_ppl = float('inf')
    opt_lambda = None
    for lambda1 in lambdas1:
        for lambda2 in lambdas2:
            if lambda1+lambda2 < 1:
                print "lambda1 " + str(lambda1) + ", lambda2 " + str(lambda2)
                perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, lambda1, lambda2)
                if perplexity < min_ppl:
                    min_ppl = perplexity
                    opt_lambda = (lambda1,lambda2)
                print "#perplexity: " + str(perplexity)
    print 'best perplexity: '+ str(min_ppl)
    print 'best params: l1=' + str(opt_lambda[0]) + ' l2: ' + str(opt_lambda[1])

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
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.5)
    print "#perplexity: " + str(perplexity)
    ### YOUR CODE HERE
    ### END YOUR CODE

if __name__ == "__main__":
    # test_ngram()
    grid_search()