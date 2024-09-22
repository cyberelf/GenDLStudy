# PixelCNN MD for color images

from matplotlib.pyplot import step
import numpy as np
import json
import re
import string
import pandas as pd

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, callbacks
from tensorflow.keras.utils import plot_model
import tensorflow_probability as tfp


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


IMAGE_SIZE = 32
N_COMPONENTS = 5
EPOCHS = 10
BATCH_SIZE = 128


# Define a Pixel CNN network
dist = tfp.distributions.PixelCNN(
    image_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
    num_resnet=1,
    num_hierarchies=2,
    num_filters=32,
    num_logistic_mix=N_COMPONENTS,
    dropout_p=0.3,
)

# Define the model input
image_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

# Define the log likelihood for the loss fn
log_prob = dist.log_prob(image_input)

# Define the model
pixelcnn = models.Model(inputs=image_input, outputs=log_prob)
pixelcnn.add_loss(-tf.reduce_mean(log_prob))

# Compile and train the model
pixelcnn.compile(
    optimizer=optimizers.Adam(0.001),
)

pixelcnn.summary()
plot_model(pixelcnn, to_file='model++.png', show_shapes=True, show_layer_names=True)


# prepare the data
(x_train, _), (_, _) = datasets.fashion_mnist.load_data()

# Preprocess the data
def preprocess(imgs):
    imgs = np.expand_dims(imgs, -1)
    imgs = tf.image.resize(imgs, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
    return imgs


input_data = preprocess(x_train)


# Show some items of clothing from the training set
display(input_data)


# Train the model
tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")


class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img):
        self.num_img = num_img

    def generate(self):
        return dist.sample(self.num_img).numpy()

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.generate()
        display(
            generated_images,
            n=self.num_img,
            save_to=f"{output_dir}/generated_img_%03d.png" % (epoch),
        )


img_generator_callback = ImageGenerator(num_img=2)

pixelcnn.fit(
    input_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=True,
    callbacks=[tensorboard_callback, img_generator_callback],
)

generated_images = img_generator_callback.generate()

display(generated_images, n=img_generator_callback.num_img)
