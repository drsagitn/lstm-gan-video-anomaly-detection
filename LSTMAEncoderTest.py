# Basic libraries
import numpy as np
import tensorflow as tf
from data_gen import get_train_data, get_next_batch, save_images
import os

tf.reset_default_graph()
tf.set_random_seed(2016)
np.random.seed(2016)

# LSTM-autoencoder
from LSTMAutoencoder import *

# Constants
batch_size = 2
hidden_num = 1024
step_num = 10 # number of frames per time
elem_num = 115*76  # frame size h x w
n_epoch = 100

# placeholder list
p_input = tf.placeholder(tf.float32, shape=(batch_size, step_num, elem_num))
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]

cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    save_model = "models/test.ckpt"
    if os.path.exists(save_model + ".meta"):
        print("load save model")
        saver.restore(sess, save_model)
        XTest = get_train_data("data/scaled_data/UCSDped1/Train", step_num)

        iteration = 1

        for i in range(iteration):
            pinput = get_next_batch(XTest, i+15, batch_size)

            (input_, output_) = sess.run([ae.input_, ae.output_], {p_input: pinput})

            for j in range(batch_size):
                save_images(output_[j, :, :], "output_" + str(i) + "_" + str(j))
                save_images(input_[j, :, :], "input_" + str(i) + "_" + str(j))

        print('train result :')
        print('input :', input_[0, :, :].flatten())
        print('input size: ',input_[0, :, :].flatten().shape)
        print('output :', output_[0, :, :].flatten())
        print('output2 :', output_[1, :, :].flatten())
        print('diff value :', np.sum(np.absolute(input_[0, :, :].flatten() - output_[0, :, :].flatten())))
    else:
        print("No saved model found!")


