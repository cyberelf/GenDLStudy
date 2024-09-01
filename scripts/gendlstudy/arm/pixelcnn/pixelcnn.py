#!/usr/bin/env python
# coding: utf-8

# # 👾 PixelCNN from scratch

# In this notebook, we'll walk through the steps required to train your own PixelCNN on the fashion MNIST dataset from scratch

# The code has been adapted from the excellent [PixelCNN tutorial](https://keras.io/examples/generative/pixelcnn/) created by ADMoreau, available on the Keras website.


from matplotlib.pyplot import step
import numpy as np
import json
import re
import string
import pandas as pd

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, callbacks
from tensorflow.keras.utils import plot_model


import os
from gendlstudy.utils import sample_batch, display
from gendlstudy.arm.lstm.manual_lstm import LSTMModel

data_dir = "/gemini/code/GenDLStudy/data/epirecipes"
base_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = f"{base_dir}/checkpoint"
output_dir = f"{base_dir}/output"
log_dir = f"{base_dir}/logs"
model_dir = f"{base_dir}/models"

timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
log_dir = f"{log_dir}/{timestamp}"

for dir in [checkpoint_dir, output_dir, log_dir, model_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# ## 0. Parameters <a name="parameters"></a>

IMAGE_SIZE = 16
PIXEL_LEVELS = 4
N_FILTERS = 128
RESIDUAL_BLOCKS = 5
BATCH_SIZE = 128
EPOCHS = 150

# ## 1. Prepare the data <a name="prepare"></a>

# Load the data
(x_train, _), (_, _) = datasets.fashion_mnist.load_data()


# Preprocess the data
def preprocess(imgs_int):
    imgs_int = np.expand_dims(imgs_int, -1)
    imgs_int = tf.image.resize(imgs_int, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
    imgs_int = (imgs_int / (256 / PIXEL_LEVELS)).astype(int)
    imgs = imgs_int.astype("float32")
    imgs = imgs / PIXEL_LEVELS
    return imgs, imgs_int


input_data, output_data = preprocess(x_train)


# Show some items of clothing from the training set
display(input_data)


# ## 2. Build the PixelCNN

# The first layer is the PixelCNN layer. This layer simply
# builds on the 2D convolutional layer, but includes masking.
class MaskedConv2D(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(MaskedConv2D, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

    def get_config(self):
        cfg = super().get_config()
        return cfg


class ResidualBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        # why use half of the filters?
        self.conv1 = layers.Conv2D(
            filters=filters // 2, kernel_size=1, activation="relu"
        )
        self.pixel_conv = MaskedConv2D(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return layers.add([inputs, x])

    def get_config(self):
        cfg = super().get_config()
        return cfg


inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
# 第一个MaskedConv2D采用A的模式，掩住当前点
x = MaskedConv2D(
    mask_type="A",
    filters=N_FILTERS,
    kernel_size=7,
    activation="relu",
    padding="same",
)(inputs)

# 前面kernel size 7的MaskedConv2D, 之后是5个residual block, 每个block有一个kernel size 3的MaskedConv2D
# 这样相当于每个点可以看到上面和左面3+1*5=8个点的信息，刚好可以覆盖16*16的image
# 残差连接是为了保留原始的输入信息，辅助模型训练
for _ in range(RESIDUAL_BLOCKS):
    # x = ResidualBlock(filters=N_FILTERS)(x)
    ix = layers.Conv2D(
        filters=N_FILTERS // 2, kernel_size=1, activation="relu"
    )(x)
    ix = MaskedConv2D(
        mask_type="B",
        filters=N_FILTERS // 2,
        kernel_size=3,
        activation="relu",
        padding="same",
    )(ix)
    ix = layers.Conv2D(
        filters=N_FILTERS, kernel_size=1, activation="relu"
    )(ix)
    x = layers.add([x, ix])

# Two more MaskedConv2D layers, what is the purpose of these?
# It seem redundant to me, using normal Conv2D layers would have been enough
for _ in range(2):
    x = MaskedConv2D(
        mask_type="B",
        filters=N_FILTERS,
        kernel_size=1,
        strides=1,
        activation="relu",
        padding="valid",
    )(x)

# Size one kernel to reduce the channel to 4 gray scale levels
out = layers.Conv2D(
    filters=PIXEL_LEVELS,
    kernel_size=1,
    strides=1,
    activation="softmax",
    padding="valid",
)(x)

pixel_cnn = models.Model(inputs, out)
pixel_cnn.summary()
plot_model(pixel_cnn, to_file='model.png', show_shapes=True, show_layer_names=True)


# ## 3. Train the PixelCNN <a name="train"></a>

adam = optimizers.Adam(learning_rate=0.0005)
# loss is possibility of each pixel level vs the actual pixel level (0..4)
pixel_cnn.compile(optimizer=adam, loss="sparse_categorical_crossentropy")


tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir)


class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img):
        self.num_img = num_img

    def sample_from(self, probs, temperature):  # <2>
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs)

    def generate(self, temperature):
        # empty image in the beginning 
        generated_images = np.zeros(
            shape=(self.num_img,) + (pixel_cnn.input_shape)[1:]
        )
        batch, rows, cols, channels = generated_images.shape

        for row in range(rows):
            for col in range(cols):
                for channel in range(channels):
                    # generate pixel by pixel
                    # the probs has 4 filters representing probabilities of the gray scale level of each pixel
                    probs = self.model.predict(generated_images, verbose=0)[
                        :, row, col, :
                    ]
                    # sample one from the 4 channels with temperature
                    generated_images[:, row, col, channel] = [
                        self.sample_from(x, temperature) for x in probs
                    ]
                    # normalize the pixel value
                    generated_images[:, row, col, channel] /= PIXEL_LEVELS

        return generated_images

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.generate(temperature=1.0)
        display(
            generated_images,
            save_to=f"{output_dir}/generated_img_%03d.png" % (epoch),
        )


img_generator_callback = ImageGenerator(num_img=10)


pixel_cnn.fit(
    input_data,
    output_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[tensorboard_callback, img_generator_callback],
)


# ## 4. Generate images <a name="generate"></a>

generated_images = img_generator_callback.generate(temperature=1.0)


display(generated_images)

