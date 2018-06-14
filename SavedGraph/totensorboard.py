# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:53:08 2018

@author: tcmxx
"""

import tensorflow as tf
import os


def create_recurrent_encoder(name='lstm'):

    input_state = tf.placeholder(shape=[None, 25], dtype=tf.float32, name='test')
    memory_in = tf.placeholder(shape=[None, 14], dtype=tf.float32, name='testmemornyin')

    s_size = input_state.get_shape().as_list()[1]
    m_size = memory_in.get_shape().as_list()[1]
    lstm_input_state = tf.reshape(input_state, shape=[-1, 64, s_size])
    memory_in = tf.reshape(memory_in[:, :], [-1, m_size])
    _half_point = int(m_size / 2)
    with tf.variable_scope(name):
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(_half_point)
        lstm_vector_in = tf.nn.rnn_cell.LSTMStateTuple(memory_in[:, :_half_point], memory_in[:, _half_point:])
        recurrent_output, lstm_state_out = tf.nn.dynamic_rnn(rnn_cell, lstm_input_state,
                                                                initial_state=lstm_vector_in)

    recurrent_output = tf.reshape(recurrent_output, shape=[-1, _half_point])
    recurrent_output = tf.identity(recurrent_output,name='result')
    return recurrent_output, tf.concat([lstm_state_out.c, lstm_state_out.h], axis=1)


tf.reset_default_graph() #Clear the Tensorflow graph.x

with tf.Session() as sess:

    with tf.gfile.FastGFile("PPOTest.pb",'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        create_recurrent_encoder()
        #%%#######################################3
        #write the summary for tensorboard
        train_writer = tf.summary.FileWriter(os.getcwd() + '/logs', sess.graph)
        train_writer.close()
