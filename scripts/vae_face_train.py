from pathlib import Path
import tensorflow as tf
import numpy as np

import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from keras import callbacks, layers, losses, metrics, models, utils

IMAGE_SIZE = 64
BATCH_SIZE = 128
Z_DIM = 200  # dimenstions of embedding space
# VALIDATION_SPLIT = 0.2
EPOCHS = 10
CHANNELS = 3

DATAB = '../data'
DATAF = f'{DATAB}/img_align_celeba'
MOD_NAME = 'vae-face.keras'


def preprocess_img(img):
    img = tf.cast(img, tf.float32) / 255.0
    return img


@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@keras.saving.register_keras_serializable()
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstructionk_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(500 * losses.binary_crossentropy(data, reconstruction, axis=(1,2,3)))
            kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        base_config = super().get_config()
        config = {
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "decoder": keras.saving.serialize_keras_object(self.decoder),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop("encoder")
        encoder = keras.saving.deserialize_keras_object(encoder_config)
        decoder_config = config.pop("decoder")
        decoder = keras.saving.deserialize_keras_object(decoder_config)
        return cls(encoder, decoder, **config)


class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img, latent_dim):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(
            shape=(self.num_img, self.latent_dim)
        )
        generated_images = self.model.decoder(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = utils.array_to_img(generated_images[i])
            img.save("./output/generated_img_%03d_%d.png" % (epoch, i))


if __name__ == "__main__":
    img_no = get_ipython().getoutput(f'ls {DATAF} | wc -l')
    print(f'There are {img_no[0]} image fiies in folder /content/img_align_celeb.')

    train_data = utils.image_dataset_from_directory(
        DATAF,
        labels=None,
        color_mode='rgb',
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        interpolation='bilinear',)

    train = train_data.map(lambda x: preprocess_img(x))


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


    decoder_input = layers.Input(shape=(Z_DIM, ), name='decode_input')
    x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape(shape_before_flattening)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    decoder_output = layers.Conv2D(CHANNELS, (3, 3), strides=1, activation='sigmoid', padding='same', name='decoder_output')(x)
    decoder = models.Model(decoder_input, decoder_output, name='decoder')
    decoder.summary()


    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam())


    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath="./checkpoint/vae.keras",
        save_weights_only=False,
        save_freq="epoch",
        monitor="loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    )

    tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")

    get_ipython().system('mkdir -p ./output')
    vae.fit(
        train,
        # epochs=EPOCHS,
        epochs=1,
        callbacks=[
            model_checkpoint_callback,
            tensorboard_callback,
            ImageGenerator(num_img=10, latent_dim=Z_DIM),
        ],
    )

    print(vae.summary())
    vae.save(MOD_NAME)
