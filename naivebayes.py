######################################################
# NAIVE BAYES
# Zi Mo Su
######################################################

import os
import numpy as np
from numpy import *
import re
import time


######################################################
# FUNCTIONS
######################################################

def preprocess(line):
    """
    This function preprocesses a single line in a file by removing punctuation and extra spaces.
    :param line: (string) line in file
    :return: (string) processed line
    """
    # remove punctuation
    line = re.sub(r'(\W)', ' ', line)

    # remove extra spaces including at beginning and ending of sentence
    line = re.sub(r'\s+', ' ', line)
    line = re.sub(r'\s$', '', line)
    line = re.sub(r'^\s', '', line)

    # make all words lowercase
    line = line.lower()

    return line


def partition(dir):
    """
    This function partitions all reviews in a directory into a training set (size 800), test set (size 100) and
    validation set (size 100).
    :param dir: (string) name of directory
    :return: (lists) lists of strings containing reviews for each set
    """
    # random permutation
    rand = np.random.permutation(1000)
    data = []

    # loop through all files
    for filename in os.listdir(dir):
        input_file = open(os.path.join(dir, filename))

        # process each line in current file and append as string
        processed = ''
        for line in input_file:
            processed += ' ' + preprocess(line)

        data += [processed]
        input_file.close()

    data = array(data)

    # partition into test, validation and training set using random permutation
    test = data[rand[:100]]
    val = data[rand[100:200]]
    train = data[rand[200:]]

    return test, val, train


def get_counts(train):
    """
    This function gets the number of times a word is present in a review for the class defined by the training
    set input.
    :param train: (list) training set
    :return: (dict) counts is a dictionary with keys that are strings (words) and values that are their floats (count)
             (float) total count
    """
    counts = {}
    total_count = 0.0

    for review in train:
        counted = []
        words = review.split()
        for word in words:
            if word in counted:
                continue

            if word not in counts:
                counts[word] = 0

            counts[word] += 1.0/len(words)
            total_count += 1.0/len(words)

            counted += [word]

    return counts, total_count


def classify(review, neg_probs, pos_probs, neg_total, pos_total, m, k):
    """
    This function classifies a review as either positive or negative.
    :param review: (string) a review
    :param neg_probs: (dict) probabilities generated from negative reviews
    :param pos_probs: (dict) probabilities generated from positive reviews
    :param neg_total: (float) total count for negative reviews
    :param pos_total: (float) total count for positive reviews
    :param m: (float) tunable parameter for smoothing
    :param k: (float) tunable parameter for smoothing
    :return: (string) 'neg' if classified as negative, 'pos' if classified as positive
    """
    neg_prob = 0.0
    pos_prob = 0.0

    for word in review.split():
        if word in neg_probs:
            neg_prob += log(neg_probs[word])
        else:
            neg_prob += log((m * k)/(neg_total + k))

        if word in pos_probs:
            pos_prob += log(pos_probs[word])
        else:
            pos_prob += log((m * k)/(pos_total + k))

    # print 'neg:', neg_prob
    # print 'pos:', pos_prob

    # final probability [product of conditionals P(word|class)] * [P(class)]
    neg_prob = neg_prob + log(1/2.0)
    pos_prob = pos_prob + log(1/2.0)

    if neg_prob > pos_prob:
        return 'neg'
    else:
        return 'pos'


def get_performance(neg_probs, pos_probs, neg_val, pos_val, neg_total, pos_total, m, k):
    """
    This function gets the performance of the classifier on a set.
    :param neg_probs: (dict) probabilities generated from negative reviews
    :param pos_probs: (dict) probabilities generated from positive reviews
    :param neg_val: (list) list of negative reviews to be classified
    :param pos_val: (list) list of positive reviews to be classified
    :param neg_total: (float) total count for negative reviews
    :param pos_total: (float) total count for positive reviews
    :param m: (float) tunable parameter for smoothing
    :param k: (float) tunable parameter for smoothing
    :return: (floats) performance for negative, positive and total
    """
    neg_correct = 0.0
    neg_wrong = 0.0
    pos_correct = 0.0
    pos_wrong = 0.0

    for review in neg_val:
        c = classify(review, neg_probs, pos_probs, neg_total, pos_total, m, k)
        if c == 'neg':
            neg_correct += 1
        else:
            neg_wrong += 1

    for review in pos_val:
        c = classify(review, neg_probs, pos_probs, neg_total, pos_total, m, k)
        if c == 'pos':
            pos_correct += 1
        else:
            pos_wrong += 1

    neg_perf = neg_correct/(neg_correct + neg_wrong)
    pos_perf = pos_correct/(pos_correct + pos_wrong)

    perf = (neg_correct + pos_correct)/(neg_correct + neg_wrong + pos_correct + pos_wrong)

    return neg_perf, pos_perf, perf


def find_best_params(m_s, k_s, neg_train, pos_train, neg_val, pos_val):
    """
    This function finds the best m's and k's in terms of performance.
    :param m_s: (list) list of m values to test
    :param k_s: (list) list of k values to test
    :return: (floats) best m and k value
    """
    best_perf = 0

    neg_counts, neg_total = get_counts(neg_train)
    pos_counts, pos_total = get_counts(pos_train)

    for m in m_s:
        for k in k_s:
            neg_probs = {key: (val + m * k)/(neg_total + k) for key, val in neg_counts.items()}
            pos_probs = {key: (val + m * k)/(pos_total + k) for key, val in pos_counts.items()}

            neg_perf, pos_perf, perf = get_performance(neg_probs, pos_probs, neg_val, pos_val, neg_total, pos_total, m, k)

            print 'Test using m =', m, 'and k =', k
            print 'Classification performance:'
            print '\tNegative:', neg_perf
            print '\tPositive:', pos_perf
            print '\tTotal:', perf
            print

            if perf > best_perf:
                best_perf = perf
                best_m = m
                best_k = k

    print 'Best parameters found to be m =', best_m, 'and k =', best_k, 'with performance of', best_perf
    print

    return best_m, best_k


def get_top10(neg_counts, pos_counts):
    """
    This function gets the top 10 highest frequency words in the negative and positive sets that are exclusive to their
    own set.
    :param neg_counts: (dict) counts generated from negative reviews
    :param pos_counts: (dict) counts generated from positive reviews
    :return: (lists) top 10 highest frequency words for negative and positive reviews
    """
    neg_words = sorted(neg_counts, key=neg_counts.get, reverse=True)
    pos_words = sorted(pos_counts, key=pos_counts.get, reverse=True)

    neg_top10 = []
    count = 0
    for word in neg_words:
        if count > 10:
            break
        if word not in pos_words:
            neg_top10 += [word]
            count += 1

    pos_top10 = []
    count = 0
    for word in pos_words:
        if count > 10:
            break
        if word not in neg_words:
            pos_top10 += [word]
            count += 1

    return neg_top10, pos_top10

# #computing the three most frequenct words:
# def freq_words(pos_train,neg_train):
#     pos_words = []
#     neg_words = []
#     counts_pdic = get_counts(pos_train)[0]
#     counts_ndic = get_counts(neg_train)[0]
#     #obtain the intersection between the two dictionaries and delete them
#     #as we are interested in the unique words that evaluate them as positive
#     #or negative
#     intersection = counts_pdic.viewkeys() & counts_ndic.viewkeys()
#     for word in intersection:
#         #print(word)
#         if word in counts_pdic:
#             counts_pdic.pop(word)
#         if word in counts_ndic:
#             counts_ndic.pop(word)
#
#     #arrange the count values from highest to lowest
#     counts_pval = sorted(counts_pdic.values(), key=float, reverse=True)
#     counts_nval = sorted(counts_ndic.values(), key=float, reverse=True)
#     for i in range(len(counts_pval)):
#         pword = counts_pdic.keys()[list(counts_pdic.values()).index(counts_pval[i])]
#         nword = counts_ndic.keys()[list(counts_ndic.values()).index(counts_nval[i])]
#         #delete every word that will be appended as some count
#         #values may be the same
#         counts_pdic.pop(pword)
#         counts_ndic.pop(nword)
#         pos_words.append(pword+' '+str(counts_pval[i]))
#         neg_words.append(nword+' '+str(counts_nval[i]))
#         #stop when we have the first 10 values
#         if len(pos_words) == 10 and len(neg_words) == 10:
#             break
#     return pos_words,neg_words

######################################################
# MAIN CODE
######################################################

if __name__ == '__main__':

    # generate random seed
    t = int(time.time())
    t = 1489990171
    print "t =", t
    random.seed(t)

    neg_dir = 'review_polarity/txt_sentoken/neg'
    pos_dir = 'review_polarity/txt_sentoken/pos'

    neg_test, neg_val, neg_train = partition(neg_dir)
    pos_test, pos_val, pos_train = partition(pos_dir)

    m_s = [0.000001, 0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1]
    k_s = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1, 1.5, 2]

    best_m, best_k = find_best_params(m_s, k_s, neg_train, pos_train, neg_val, pos_val)

    neg_counts, neg_total = get_counts(neg_train)
    pos_counts, pos_total = get_counts(pos_train)

    neg_probs = {key: (val + best_m * best_k)/(neg_total + best_k) for key, val in neg_counts.items()}
    pos_probs = {key: (val + best_m * best_k)/(pos_total + best_k) for key, val in pos_counts.items()}

    neg_perf, pos_perf, perf = get_performance(neg_probs, pos_probs, neg_test, pos_test, neg_total, pos_total, best_m, best_k)
    print 'Using the best parameters, the performance on the test set is:'
    print '\tNegative:', neg_perf
    print '\tPositive:', pos_perf
    print '\tTotal:', perf
    print

    neg_perf, pos_perf, perf = get_performance(neg_probs, pos_probs, neg_train, pos_train, neg_total, pos_total, best_m, best_k)
    print 'Using the best parameters, the performance on the training set is:'
    print '\tNegative:', neg_perf
    print '\tPositive:', pos_perf
    print '\tTotal:', perf
    print

    neg_top10, pos_top10 = get_top10(neg_counts, pos_counts)

    print 'The top 10 words for determining a negative review are:', neg_top10
    print 'The top 10 words for determining a positive review are:', pos_top10


