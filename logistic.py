######################################################
# LOGISTIC
# Zi Mo Su
######################################################

import os
import numpy as np
from numpy import *
import re
import tensorflow as tf
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

    # remove extra spaces including at beginning and end of sentence
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


def get_unique_words(train):
    """
    This function gets the list of unique words given a training set.
    :param train: (list) training set
    :return: (list) list of unique words
    """
    unique_words = []

    for review in train:
        words = review.split()

        for word in words:
            if word in unique_words:
                continue

            unique_words += [word]

    return unique_words


def get_input(set, unique_words):
    """
    This function gets the inputs to the neural network.
    :param set: (list) training, test or validation set
    :param unique_words: (list) list of unique words in the training set
    :return: (np arrays) x and y_ inputs generated from set
    """
    total_count = len(unique_words)
    set_x = zeros((0, total_count))
    set_length = len(set)

    for review in set:
        vector = zeros((1, total_count))
        words = review.split()

        for word in words:
            if word in unique_words:
                vector[:, unique_words.index(word)] = 1

        set_x = vstack((set_x, vector))

    set_y_ = vstack((ones((set_length/2, 1)), zeros((set_length/2, 1))))

    return set_x, set_y_


def create_nn(total_count, lam):
    """
    This function creates a neural network for a logistic regression model.
    :param total_count: (int) size of input layer
    :param lam: (float) regularization parameter
    :return: neural network
    """
    # create placeholder for input
    x = tf.placeholder(tf.float32, [None, total_count])

    # output
    W0 = tf.Variable(tf.random_normal([total_count, 1], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([1], stddev=0.01))

    y = tf.nn.sigmoid(tf.matmul(x, W0)+b0)

    # create placeholder for classification input
    y_ = tf.placeholder(tf.float32, [None, 1])

    # define cost and training step
    decay_penalty = lam*tf.reduce_sum(tf.square(W0))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y)+(1-y_)*tf.log(1-y))+decay_penalty

    train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    correct_prediction = tf.equal(tf.round(y), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return sess, x, y_, train_step, decay_penalty, accuracy, W0, y


def get_best_nn(train_x, train_y_, val_x, val_y_, total_count, lams):
    """
    This function returns the best regularization parameter lambda.
    :param train_x: (np array) training set input x
    :param train_y_: (np array) training set input y_
    :param val_x: (np array) validation set input x
    :param val_y_: (np array) validation set input y_
    :param total_count: (int) size of input layer
    :param lams: (list) list of lambdas to test
    :return: (float) lambda value that yeilds best performance on the validation set
    """
    final_max = 0
    for lam in lams:
        max_val = 0
        sess, x, y_, train_step, decay_penalty, accuracy, W0, y = create_nn(total_count, lam)
        print 'Testing with:'
        print '\tLambda', lam

        for i in range(151):
            sess.run(train_step, feed_dict={x: train_x, y_: train_y_})

            val_acc = sess.run(accuracy, feed_dict={x: val_x, y_: val_y_})
            if val_acc > max_val:
                max_val = val_acc

        if max_val > final_max:
            final_max = max_val
            best_lam = lam
        print 'Best validation performance was:', final_max
        print

    print 'The chosen lambda is:'
    print '\tLambda', best_lam
    print

    return best_lam


######################################################
# MAIN CODE
######################################################

if __name__ == '__main__':

    # generate random seed
    t = int(time.time())
    t = 1490118450
    print "t =", t
    random.seed(t)

    neg_dir = 'review_polarity/txt_sentoken/neg'
    pos_dir = 'review_polarity/txt_sentoken/pos'

    # ensure a folder called 'logistic' is created in the working directory
    if not os.path.exists('logistic/test_x.npy'):
        neg_test, neg_val, neg_train = partition(neg_dir)
        pos_test, pos_val, pos_train = partition(pos_dir)

        test = hstack((neg_test, pos_test))
        val = hstack((neg_val, pos_val))
        train = hstack((neg_train, pos_train))

        unique_words = get_unique_words(train)
        total_count = len(unique_words)

        test_x, test_y_ = get_input(test, unique_words)
        val_x, val_y_ = get_input(val, unique_words)
        train_x, train_y_ = get_input(train, unique_words)

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

    lams = [0, 0.1, 0.5, 1, 2, 5, 10, 100]

    lam = get_best_nn(train_x, train_y_, val_x, val_y_, total_count, lams)

    sess, x, y_, train_step, decay_penalty, accuracy, W0, y = create_nn(total_count, lam)

    test_plot = ''
    val_plot = ''
    train_plot = ''

    for i in range(151):
        sess.run(train_step, feed_dict={x: train_x, y_: train_y_})

        if i % 10 == 0:
            print 'i=', i

            test_acc = sess.run(accuracy, feed_dict={x: test_x, y_: test_y_})
            print 'Test:', test_acc

            val_acc = sess.run(accuracy, feed_dict={x: val_x, y_: val_y_})
            print 'Validation:', val_acc

            train_acc = sess.run(accuracy, feed_dict={x: train_x, y_: train_y_})
            print 'Train:', train_acc

            print 'Penalty:', sess.run(decay_penalty)

            test_plot += str((i, test_acc*100))
            val_plot += str((i, val_acc*100))
            train_plot += str((i, train_acc*100))

    print
    print 'Output for LaTeX plotting:'
    print 'Test', test_plot
    print 'Validation', val_plot
    print 'Train', train_plot


