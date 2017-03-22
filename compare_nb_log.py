######################################################
# COMPARE NAIVE BAYES TO LOGISTIC
# Zi Mo Su
######################################################

import os
import numpy as np
from numpy import *
import re
import tensorflow as tf
import time
import naivebayes as nb
import logistic as lg


######################################################
# FUNCTIONS
######################################################

def get_largest_thetas(thetas, unique_words):
    """
    This function gets the 100 largest values in thetas.
    :param thetas: (iterable) list of theta values
    :param unique_words: (list) list of unique words in the training set
    :return: (list) 100 words corresponding to the largest 100 thetas
    """
    largest_thetas = []

    for _ in range(100):
        max_index = argmax(thetas)
        thetas[max_index] = -float('inf')
        largest_thetas += [unique_words[max_index]]

    return largest_thetas


######################################################
# MAIN CODE
######################################################

if __name__ == '__main__':

    # generate random seed
    t = int(time.time())
    t = 1490118450  # used to generate logistic data sets
    print "t =", t
    random.seed(t)

    neg_dir = 'review_polarity/txt_sentoken/neg'
    pos_dir = 'review_polarity/txt_sentoken/pos'

    neg_test, neg_val, neg_train = nb.partition(neg_dir)
    pos_test, pos_val, pos_train = nb.partition(pos_dir)

    # ensure a folder called 'logistic' is created in the working directory
    if not os.path.exists('logistic/test_x.npy'):
        test = hstack((neg_test, pos_test))
        val = hstack((neg_val, pos_val))
        train = hstack((neg_train, pos_train))

        unique_words = lg.get_unique_words(train)
        total_count = len(unique_words)

        test_x, test_y_ = lg.get_input(test, unique_words)
        val_x, val_y_ = lg.get_input(val, unique_words)
        train_x, train_y_ = lg.get_input(train, unique_words)

        np.save('logistic/test_x.npy', test_x)
        np.save('logistic/test_y_.npy', test_y_)
        np.save('logistic/val_x.npy', val_x)
        np.save('logistic/val_y_.npy', val_y_)
        np.save('logistic/train_x.npy', train_x)
        np.save('logistic/train_y_.npy', train_y_)
        np.save('logistic/unique_words.npy', unique_words)
        np.save('logistic/total_count.npy', array([total_count]))

    else:
        test_x = np.load('logistic/test_x.npy')
        test_y_ = np.load('logistic/test_y_.npy')
        val_x = np.load('logistic/val_x.npy')
        val_y_ = np.load('logistic/val_y_.npy')
        train_x = np.load('logistic/train_x.npy')
        train_y_ = np.load('logistic/train_y_.npy')
        unique_words = np.load('logistic/unique_words.npy')
        total_count = np.load('logistic/total_count.npy')[0]


    # Naive Bayes

    m_s = [0.0005]
    k_s = [1.5]

    best_m, best_k = nb.find_best_params(m_s, k_s, neg_train, pos_train, neg_val, pos_val)

    neg_counts, neg_total = nb.get_counts(neg_train)
    pos_counts, pos_total = nb.get_counts(pos_train)

    neg_probs = {key: (val + best_m * best_k)/(neg_total + best_k) for key, val in neg_counts.items()}
    pos_probs = {key: (val + best_m * best_k)/(pos_total + best_k) for key, val in pos_counts.items()}

    thetas = []
    for word in unique_words:
        if word in neg_probs and word in pos_probs:
            thetas += [log(neg_probs[word]/pos_probs[word])]

        else:
            thetas += [-float('inf')]

    nb_largest_thetas = get_largest_thetas(thetas, unique_words)


    # Logistic Regression

    lams = [10]

    lam = lg.get_best_nn(train_x, train_y_, val_x, val_y_, total_count, lams)

    sess, x, y_, train_step, decay_penalty, accuracy, W0, y = lg.create_nn(total_count, lam)

    for i in range(151):
        sess.run(train_step, feed_dict={x: train_x, y_: train_y_})

        if i % 10 == 0:
            print "i=", i

            test_acc = sess.run(accuracy, feed_dict={x: test_x, y_: test_y_})
            print "Test:", test_acc

            val_acc = sess.run(accuracy, feed_dict={x: val_x, y_: val_y_})
            print "Validation:", val_acc

            train_acc = sess.run(accuracy, feed_dict={x: train_x, y_: train_y_})
            print "Train:", train_acc

            print "Penalty:", sess.run(decay_penalty)

    thetas = sess.run(W0)

    lg_largest_thetas = get_largest_thetas(thetas, unique_words)


    # Compare Naive Bayes to Logistic Regression

    print nb_largest_thetas
    print lg_largest_thetas
