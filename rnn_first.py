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
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import BasicLSTMCell,static_rnn

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
headers = ["Praise"]

df_y = df.loc[:, headers]
df_y.head()
df_y[df_y != 0] = 1
df_y = df_y.round(0).astype(int)
df_y['new'] = 1 - df_y
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
chunk_size=numFeatures/10
n_chunks=numFeatures/chunk_size
n_classes = 2
rnn_size = numFeatures

x = tf.placeholder('float', [None, 1001,1])
y = tf.placeholder('float')

def recurrent_neural_network_model(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}

    lstm_cell=BasicLSTMCell(rnn_size)
    lstm_cell1=BasicLSTMCell(rnn_size)

    outputs,states=tf.nn.bidirectional_dynamic_rnn(lstm_cell,lstm_cell1,x,dtype=tf.float32)
    output = tf.matmul(outputs[-1][-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    display_step = 50
    for i in range(numEpochs):
        print i
        _, c = sess.run([optimizer, cost], feed_dict={x: inputX.reshape(-1,1001,1), y: inputY})
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc,c = sess.run([accuracy, cost], feed_dict={x: inputX.reshape(-1,1001,1),y: inputY})
        if i % display_step == 0:
            print('Accuracy:',acc)
    print("Accuracy:", \
        sess.run(accuracy, feed_dict={x: inputX.reshape(-1,1001,1),
                                      y: inputY,
                                      }))
print(inputX.shape)
print(inputY.shape)
print(outputX.shape)
print(outputY.shape)
train_neural_network(x)