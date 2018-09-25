import tensorflow as tf
import GanArch
from util import is_existing
from LSTMAutoencoder import LSTMAutoencoder
import numpy as np
import time
from data_gen import get_train_data, get_test_data


def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter

# tf.reset_default_graph()
with tf.Graph().as_default():
    tf.set_random_seed(42)

    #TRAINING PARAMETERS
    nb_epochs = 1
    method = "cross-e"  # cross-e or fm  method for discrimination loss
    weight = 0.1  # weight for sum of mapping loss function
    degree = 1  # degree for L norm


    # LSTM AE parameters
    hidden_num = 784 # number of neutron in a LSTM cell
    batch_num = 1
    step_num = 20 # number of step in LSTM network, usually number of frames in video  200
    elem_num = 400 # number of input for each step, usually number of pixels in one frame  37604

    p_input = tf.placeholder(tf.float32, shape=(batch_num, step_num, elem_num))
    p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]
    cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
    ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=True)

    # 2 Sample random noise and generate representation with generator
    # GENERATOR INPUT Parameters
    input_dim = (None, 28, 28, 1)
    starting_lr = 0.00001
    batch_size = 2
    latent_dim = 200
    ema_decay = 0.999
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")
    gen = GanArch.generator
    dis = GanArch.discriminator
    random_z = tf.random_normal([batch_size, latent_dim], mean=0.0, stddev=1.0, name='random_z')
    gen_out = gen(random_z, is_training=is_training_pl)
    input_pl = tf.placeholder(tf.float32, shape=input_dim, name="input") #tf.reshape(ae.enc_state.c, [-1, 28, 28, 1])

    # Pass the real representation and fake one into the discriminator
    real_d, inter_layer_real = dis(input_pl, is_training=is_training_pl)
    fake_d, inter_layer_fake = dis(gen_out, is_training=is_training_pl, reuse=True)

    with tf.name_scope('loss_functions'):
        real_discriminator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(1, shape=[batch_size]), real_d,
                                                                  scope='real_discriminator_loss')
        fake_discriminator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(0, shape=[batch_size]), fake_d,
                                                                  scope='fake_discriminator_loss')
        discriminator_loss = real_discriminator_loss + fake_discriminator_loss
        generator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(1, shape=[batch_size]), fake_d,
                                                         scope='generator_loss')

    with tf.name_scope('optimizers'):
        dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        update_ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        update_ops_dis = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')

        with tf.control_dependencies(update_ops_gen):  # attached op for moving average batch norm
            gen_op = optimizer_gen.minimize(generator_loss, var_list=gvars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(discriminator_loss, var_list=dvars)

        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

    with tf.name_scope('training_summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('real_discriminator_loss', real_discriminator_loss, ['dis'])
            tf.summary.scalar('fake_discriminator_loss', fake_discriminator_loss, ['dis'])
            tf.summary.scalar('discriminator_loss', discriminator_loss, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', generator_loss, ['gen'])

        with tf.name_scope('image_summary'):
            tf.summary.image('reconstruct', gen_out, 8, ['image'])
            tf.summary.image('input_images', input_pl, 8, ['image'])


        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')
        sum_op_im = tf.summary.merge_all('image')

    with tf.variable_scope("latent_variable"):
        z_optim = tf.get_variable(name='z_optim', shape= [batch_size, latent_dim], initializer=tf.truncated_normal_initializer())
        reinit_z = z_optim.initializer
    # EMA
    generator_ema = gen(z_optim, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)
    # Pass real and fake images into discriminator separately
    real_d_ema, inter_layer_real_ema = dis(input_pl, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)
    fake_d_ema, inter_layer_fake_ema = dis(generator_ema, is_training=is_training_pl, getter=get_getter(gen_ema), reuse=True)

    with tf.name_scope('error_loss'):
        delta = input_pl - generator_ema
        delta_flat = tf.contrib.layers.flatten(delta)
        gen_score = tf.norm(delta_flat, ord=degree, axis=1, keep_dims=False, name='epsilon')

    with tf.variable_scope('Discriminator_loss'):
        if method == "cross-e":
            dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake_d_ema), logits=fake_d_ema)

        elif method == "fm":
            fm = inter_layer_real_ema - inter_layer_fake_ema
            fm = tf.contrib.layers.flatten(fm)
            dis_score = tf.norm(fm, ord=degree, axis=1, keep_dims=False,
                             name='d_loss')

        dis_score = tf.squeeze(dis_score)

    with tf.variable_scope('Total_loss'):
        loss = (1 - weight) * gen_score + weight * dis_score

    with tf.name_scope("Scores"):
        list_scores = loss

    logdir = "train_logs"

    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None,
                             save_model_secs=120)

    #data
    trainx, trainy = get_train_data()
    trainx_copy = trainx.copy()
    testx, testy = get_test_data()
    RANDOM_SEED = 146
    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    # start training
    with sv.managed_session() as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)

        train_batch = 0
        epoch = 0

        while not sv.should_stop() and epoch < nb_epochs:

            lr = starting_lr

            begin = time.time()
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling unl dataset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]

            train_loss_dis, train_loss_gen = [0, 0]
            # training
            for t in range(nr_batches_train):
                print('Step ', t, ' per ', nr_batches_train)

                # construct randomly permuted minibatches
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # train discriminator
                feed_dict = {input_pl: trainx[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
                _, ld, sm = sess.run([train_dis_op, discriminator_loss, sum_op_dis], feed_dict=feed_dict)
                train_loss_dis += ld
                writer.add_summary(sm, train_batch)

                # train generator
                feed_dict = {input_pl: trainx_copy[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
                _, lg, sm = sess.run([train_gen_op, generator_loss, sum_op_gen], feed_dict=feed_dict)
                train_loss_gen += lg
                writer.add_summary(sm, train_batch)

                if t % FREQ_PRINT == 0:  # inspect reconstruction
                    t= np.random.randint(0,4000)
                    ran_from = t
                    ran_to = t + batch_size
                    sm = sess.run(sum_op_im, feed_dict={input_pl: trainx[ran_from:ran_to],is_training_pl: False})
                    writer.add_summary(sm, train_batch)

                train_batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_dis /= nr_batches_train

            print('Epoch terminated')
            print("Epoch %d | time = %ds | loss gen = %.4f | loss dis = %.4f "
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_dis))

            epoch += 1

        inds = rng.permutation(testx.shape[0])
        testx = testx[inds]  # shuffling unl dataset
        testy = testy[inds]
        scores = []
        inference_time = []

    # with tf.Session() as sess:
    #     saver = tf.train.Saver()
    #     model_name = "test.ckpt"
    #     if is_existing(model_name):
    #         saver.restore(sess, "models/" + model_name)
    #         print("Loaded params for encoder")
    #
    #     test_sequence = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape([1,8,1])
    #     (input_, output_, enc_out) = sess.run([ae.input_, ae.output_, ae.enc_state], {p_input: test_sequence})
    #     print('input :', input_[0, :, :].flatten())
    #     print('output :', output_[0, :, :].flatten())

