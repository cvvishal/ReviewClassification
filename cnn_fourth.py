from __future__ import division
import tarfile
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd
import re
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import random
from sklearn import metrics
df = pd.read_csv("ldata.csv")

headers = list(df.columns.values)

df = df[df.Praise.notnull()]
df = df[df.Problem.notnull()]
df = df[df.Mitigation.notnull()]
df = df[df.Summary.notnull()]
df = df[df.Solution.notnull()]
df = df[df.Neutrality.notnull()]
df = df[df.Localization.notnull()]

df_x = df.loc[:, ['Comments']]

headers.remove('Comments')
headers = ["Localization"]

df_y = df.loc[:, headers]
df_y.head()
df_y[df_y != 0] = 1
df_y = df_y.round(0).astype(int)
df_y['new'] = 1 - df_y
#load model
model = Doc2Vec.load("comments2vec.d2v")


comments = []
for index, row in df.iterrows():
    line = row["Comments"]
    line = re.sub("[^a-zA-Z?!]"," ", line)
    words = [w.lower().decode('utf-8') for w in line.strip().split() if len(w)>=3]
    comments.append(words)
x_train = []
for comment in comments:
        feature_vec = model.infer_vector(comment)
        x_train.append(feature_vec)


x_test = x_train[len(x_train)-100:len(x_train)]
x_train = x_train[0:len(x_train)-100]
y_train = df_y[0:len(x_train)]
y_test = df_y[len(x_train):]
inputX = np.array(x_train)
inputY = y_train.as_matrix()
outputX = np.array(x_test)
outputY = y_test.as_matrix()
numFeatures = inputX.shape[1]
numEpochs  = 1000
n_classes = 2

x = tf.placeholder('float', [None, numFeatures])
y = tf.placeholder('float')
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_conv3':tf.Variable(tf.random_normal([5,5,64,128])),
               'W_fc':tf.Variable(tf.random_normal([5*5*128,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_conv3':tf.Variable(tf.random_normal([128])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 1, numFeatures, 1])
    conv1 = tf.nn.sigmoid(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    conv2 = tf.nn.sigmoid(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv3 = tf.nn.sigmoid(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
    fc = tf.reshape(conv3, [-1, weights['W_fc'].get_shape().as_list()[0]])
    fc = tf.nn.sigmoid(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)
    output = tf.matmul(fc, weights['out'])+biases['out']
    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( -tf.reduce_sum(y*tf.log(prediction)))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    display_step = 50
    for i in range(numEpochs):
        print i
        sess.run(optimizer, feed_dict={x: inputX, y: inputY}) 
        acc,c,pre = sess.run([accuracy, cost,correct_pred], feed_dict={x: inputX,y: inputY})
        if i % display_step == 0:
            print('Accuracy:',acc)
    print("Testing Accuracy:",sess.run(accuracy, feed_dict={x: outputX,y: outputY,}))
train_neural_network(x)