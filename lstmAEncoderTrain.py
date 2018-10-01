# Basic libraries
import numpy as np
import tensorflow as tf
from data_gen import get_train_data, get_next_batch
import os

tf.reset_default_graph()
tf.set_random_seed(2016)
np.random.seed(2016)

# LSTM-autoencoder
from LSTMAutoencoder import *

# Constants
batch_num = 2
hidden_num = 1024
step_num = 10 # number of frames per time
elem_num = 115*76  # frame size h x w
n_epoch = 1

# placeholder list
with tf.name_scope("variables_scope"):
    p_input = tf.placeholder(tf.float32, shape=(batch_num, step_num, elem_num))
    p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]
cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("train_logs/lstmae", sess.graph)
    saver = tf.train.Saver()
    save_model = "models/test.ckpt"
    if os.path.exists(save_model + ".meta"):
        print("load save model")
        saver.restore(sess, save_model)
    XTrain = get_train_data("data/scaled_data/UCSDped1/Train", step_num)

    iteration_per_epoch = int(len(XTrain)/batch_num)

    for e in range(n_epoch):
        for i in range(iteration_per_epoch):
            pinput = get_next_batch(XTrain, i, batch_num)

            (loss_val, _) = sess.run([ae.loss, ae.train], {p_input: pinput})

            print('Epoch %d ' % e, 'iter %d:' % (i + 1), loss_val)

    (input_, output_) = sess.run([ae.input_, ae.output_], {p_input: pinput})
    print('train result :')
    print('input :', input_[0, :, :].flatten())
    print('input size: ',input_[0, :, :].flatten().shape)
    print('output :', output_[0, :, :].flatten())
    print('diff value :', np.sum(np.absolute(input_[0, :, :].flatten() - output_[0, :, :].flatten())))
    save_path = saver.save(sess, "models/test.ckpt")
    print("Model saved in path: %s" % save_path)
