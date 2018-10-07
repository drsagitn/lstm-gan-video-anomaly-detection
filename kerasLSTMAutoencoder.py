from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

timesteps = 10
imgWidth = 28
imgHeight = 28
input_dim = imgWidth*imgHeight #115*76
latent_dim = 32

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

# Autoencoder
LSTMautoencoder = Model(inputs, decoded)

# Encoder model
encoder = Model(inputs, encoded)
# # Decoder model
encoder_output = Input(shape=(timesteps, latent_dim,))
decoder_layer = LSTMautoencoder.layers[-1]
decoder = Model(encoder_output, decoder_layer(encoder_output))