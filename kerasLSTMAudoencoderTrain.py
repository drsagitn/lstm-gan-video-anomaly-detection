from kerasLSTMAutoencoder import *
from keras.datasets import mnist
import numpy as np
import os
from data_gen import get_data_full

# x_train = get_data_full("data/scaled_data/UCSDped1/Train", timesteps)
# x_test = get_data_full("data/scaled_data/UCSDped1/Test", timesteps)

# from display_images import show_img, show_img_arr
# show_img_arr(x_train[2, 0:10, :].reshape(10, 76, 115), 5)

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((int(len(x_train)/timesteps), timesteps, np.prod(x_train.shape[1:])))
x_test = x_test.reshape((int(len(x_test)/timesteps), timesteps, np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

model_path = "keras_model/lstmAutoencoder.h5" #_ucsdped1
if(os.path.isfile(model_path)):
    print("Load saved model at ", model_path)
    LSTMautoencoder.load_weights(model_path)

LSTMautoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') # (optimizer='rmsprop', loss=vae_loss)
hist = LSTMautoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=200,
                shuffle=True,
                validation_data=(x_test, x_test))

if(hist.history['loss'][-1] < hist.history['loss'][0]):
    print("Model improved from ", hist.history['loss'][0], " to ", hist.history['loss'][-1])
    LSTMautoencoder.save_weights(model_path)
    print("Saved model into ", model_path)


encoded_imgs = encoder.predict(x_test)
encoded_imgs = encoded_imgs.reshape((int(len(x_test)/timesteps), timesteps, latent_dim))
decoded_imgs = decoder.predict(encoded_imgs)


import matplotlib.pyplot as plt

n = 3  # how many batch of sequence we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    img_arr = x_test[i].reshape(timesteps, imgWidth, imgHeight)
    for idx, img in enumerate(img_arr):
        ax = plt.subplot(2*n, timesteps, i*timesteps*2 + idx + 1)
        plt.imshow(img)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # display reconstruction
    img_arr = decoded_imgs[i].reshape(timesteps, imgWidth, imgHeight)
    for idx, img in enumerate(img_arr):
        ax = plt.subplot(2*n, timesteps, i*timesteps*2 + idx + 1 + timesteps)
        plt.imshow(img)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()