######################################################
# WORD2VEC
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


def get_vector(word0, word1):
    """
    This function gets the word2vec vectors from word0 and word1 and concatenates them into one 256 dimension vector.
    :param word0: (string) a word
    :param word1: (string) a word
    :return: (None or np array) None if one of the words is not in the embeddings, otherwise return the vector
    """
    global EMBEDDINGS, WORD2INDEX
    
    if word0 in WORD2INDEX and word1 in WORD2INDEX:
        vec0 = EMBEDDINGS[WORD2INDEX[word0], :]
        vec1 = EMBEDDINGS[WORD2INDEX[word1], :]
        vec = hstack((vec0, vec1))

        return vec

    else:
        return None


def get_input(set, set_name):
    """
    This function gets the inputs to the neural network.
    :param set: (list) training, test or validation set
    :param set_name: (string) name of the set
    :return: (np arrays) x and y_ inputs generated from set
    """
    global EMBEDDINGS, WORD2INDEX

    set_x = zeros((0, 256))

    for review in set:
        words = review.split()
        for i in range(len(words)-1):
            if words[i] not in WORD2INDEX or words[i+1] not in WORD2INDEX:
                continue

            x = get_vector(words[i], words[i+1])

            set_x = vstack((set_x, x))

    print 'Size of', set_name + ':', 2*set_x.shape[0]
    set_y_ = vstack((ones((set_x.shape[0], 1)), zeros((set_x.shape[0], 1))))

    rand = random.randint(0, 41524, size=(set_x.shape[0], 2))
    x = zeros(((set_x.shape[0]), 256))
    for i in range(rand.shape[0]):
        for j in range(2):
            x[i, j*128:(j+1)*128] = EMBEDDINGS[rand[i, j], :]

    set_x = vstack((set_x, x))

    return set_x, set_y_


def create_nn(lam):
    """
    This function creates a neural network for a logistic regression model.
    :param lam: (float) regularization parameter
    :return: neural network
    """
    # create placeholder for input
    x = tf.placeholder(tf.float32, [None, 256])

    # output
    W0 = tf.Variable(tf.random_normal([256, 1], stddev=0.01))
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


def get_best_nn(train_x, train_y_, val_x, val_y_, lams):
    """
    This function returns the best regularization parameter lambda.
    :param train_x: (np array) training set input x
    :param train_y_: (np array) training set input y_
    :param val_x: (np array) validation set input x
    :param val_y_: (np array) validation set input y_
    :param lams: (list) list of lambdas to test
    :return: (float) lambda value that yeilds best performance on the validation set
    """
    final_max = 0
    for lam in lams:
        max_val = 0
        sess, x, y_, train_step, decay_penalty, accuracy, W0, y = create_nn(lam)
        print 'Testing with:'
        print '\tLambda =', lam

        for i in range(301):
            sess.run(train_step, feed_dict={x: train_x, y_: train_y_})

            # print sess.run(y, feed_dict={x: val_x, y_: train_y_})

            val_acc = sess.run(accuracy, feed_dict={x: val_x, y_: val_y_})
            if val_acc > max_val:
                max_val = val_acc

            # if i % 10 == 0:
            #      print i
            #      print val_acc

        if max_val > final_max:
            final_max = max_val
            best_lam = lam
        print 'Best validation performance was:', final_max
        print

    print 'The chosen lambda is:'
    print '\tLambda =', best_lam
    print

    return best_lam


def get_closest_words(word):
    """
    This function gets the 10 closest words to the input word.
    :param word: (string) word
    :return: (list) list of 10 closest words to word
    """
    index = WORD2INDEX[word]
    vec = EMBEDDINGS[index, :]

    cos_distance = -dot(EMBEDDINGS, reshape(vec, (128, 1)))/reshape((linalg.norm(vec)*linalg.norm(EMBEDDINGS, axis=1)), (41524, 1))

    cos_distance[index] = float('inf')

    closest_words = []
    for _ in range(10):
        min_index = argmin(cos_distance)
        cos_distance[min_index, :] = float('inf')
        closest_words += [INDEX2WORD[min_index]]

    return closest_words


######################################################
# MAIN CODE
######################################################

if __name__ == '__main__':

    # generate random seed
    t = int(time.time())
    t = 1490071244  # for generating data
    print "t =", t
    random.seed(t)

    EMBEDDINGS = load('embeddings.npz')['emb']  # shape (41524, 128)
    INDEX2WORD = load('embeddings.npz')['word2ind'].flatten()[0]  # dictionary of length 41524
    WORD2INDEX = {v: k for k, v in INDEX2WORD.iteritems()}

    neg_dir = 'review_polarity/txt_sentoken/neg'
    pos_dir = 'review_polarity/txt_sentoken/pos'

    # ensure a folder called 'word2vec' is created in the working directory
    if not os.path.exists('word2vec/test_x.npy'):
        neg_test, neg_val, neg_train = partition(neg_dir)
        pos_test, pos_val, pos_train = partition(pos_dir)

        test = hstack((neg_test[:10], pos_test[:10]))
        val = hstack((neg_val[:10], pos_val[:10]))
        train = hstack((neg_train[:40], pos_train[:40]))

        test_x, test_y_ = get_input(test, 'test set')  # size: 22308
        val_x, val_y_ = get_input(val, 'validation set')  # size: 24468
        train_x, train_y_ = get_input(train, 'training set')  # size: 109144

        np.save('word2vec/test_x.npy', test_x)
        np.save('word2vec/test_y_.npy', test_y_)
        np.save('word2vec/val_x.npy', val_x)
        np.save('word2vec/val_y_.npy', val_y_)
        np.save('word2vec/train_x.npy', train_x)
        np.save('word2vec/train_y_.npy', train_y_)

    else:
        test_x = np.load('word2vec/test_x.npy')
        test_y_ = np.load('word2vec/test_y_.npy')
        val_x = np.load('word2vec/val_x.npy')
        val_y_ = np.load('word2vec/val_y_.npy')
        train_x = np.load('word2vec/train_x.npy')
        train_y_ = np.load('word2vec/train_y_.npy')

    lams = [0, 0.1, 0.5, 1, 2, 5, 10, 100]

    lam = get_best_nn(train_x, train_y_, val_x, val_y_, lams)

    sess, x, y_, train_step, decay_penalty, accuracy, W0, y = create_nn(lam)

    test_plot = ''
    val_plot = ''
    train_plot = ''

    for i in range(501):
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
    print

    # PART 8
    closest_words = get_closest_words('story')
    print 'Closest words to "story" are:', closest_words

    closest_words = get_closest_words('good')
    print 'Closest words to "good" are:', closest_words

    closest_words = get_closest_words('man')
    print 'Closest words to "man" are:', closest_words

    closest_words = get_closest_words('he')
    print 'Closest words to "he" are:', closest_words
