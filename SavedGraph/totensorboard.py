# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:53:08 2018

@author: tcmxx
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
import os


tf.reset_default_graph() #Clear the Tensorflow graph.x

with tf.Session() as sess:

    with gfile.FastGFile("test.pb",'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        
        #%%#######################################3
        #write the summary for tensorboard
        train_writer = tf.summary.FileWriter(os.getcwd() + '/logs', sess.graph)
        train_writer.close()
