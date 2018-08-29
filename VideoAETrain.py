# Basic libraries
import numpy as np
import tensorflow as tf
import os
from data_gen import get_next_batch
tf.reset_default_graph()
tf.set_random_seed(2016)
np.random.seed(2016)

# LSTM-autoencoder
from LSTMAutoencoder import *


def is_existing(model_name):
    for file in os.listdir("models"):
        if model_name in file:
            return True
    return False

# Constants
batch_num = 1
hidden_num = 128
step_num = 200  #  number of frames in video
elem_num = 37604  #  number of pixel in one frame
epochs = 10
dataset_name = 'UCSDped1'
TRAIN_DIR = 'data/' + dataset_name + '/Train'
n_train_video = len(os.listdir(TRAIN_DIR))
iter_per_epoch = int(n_train_video / batch_num)
iteration = 10000

# placeholder list
p_input = tf.placeholder(tf.float32, shape=(batch_num, step_num, elem_num))
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]

cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sequences = None
    saver = tf.train.Saver()
    model_name = "videoae_" + dataset_name + '_' + str(hidden_num) + ".ckpt"
    if is_existing(model_name):
        saver.restore(sess, "models/" + model_name)

    for i in range(epochs):
        for j in range(iter_per_epoch):
            """Random sequences.
              Every sequence has size batch_num * step_num * elem_num 
              Each step number increases 1 by 1.
              An initial number of each sequence is in the range from 0 to 19.
              (ex. [8. 9. 10. 11. 12. 13. 14. 15])
            """
            sequences = get_next_batch(j, batch_num)

            (loss_val, _) = sess.run([ae.loss, ae.train], {p_input: sequences})
            print('Epoch ', i,' iter %d:' % (j + 1), loss_val)

    (input_, output_) = sess.run([ae.input_, ae.output_], {p_input: sequences})
    print('train result :')
    print('input :', input_[0, :, :].flatten())
    print('output :', output_[0, :, :].flatten())
    print('diff value :', np.sum(input_[0, :, :].flatten() - output_[0, :, :].flatten()))

    file_path = "models/" + model_name
    save_path = saver.save(sess, file_path)
    print("Model saved in path: %s" % save_path)


