import tensorflow as tf
import tensorflow.keras.backend as K
from keras import layers, models

Z_DIM = 200  # dimenstions of embedding space
IMAGE_SIZE = 64
CHANNELS = 3

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="encoder_input")
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    shape_before_flattening = K.int_shape(x)[1:]
    print(f'K.int_shape(x): {K.int_shape(x)}')
    print(f'shape_before_flattening: {shape_before_flattening}')
    x = layers.Flatten()(x)

    z_mean = layers.Dense(Z_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(Z_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

