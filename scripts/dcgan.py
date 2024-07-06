from matplotlib import pyplot as plt
import tensorflow as tf
from keras import (
    layers,
    models,
    callbacks,
    losses,
    utils,
    metrics,
    optimizers,
)


def sample_batch(dataset):
    batch = dataset.take(1).get_single_element()
    if isinstance(batch, tuple):
        batch = batch[0]
    return batch.numpy()


def display(
    images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None
):
    """
    Displays n random images from each one of the supplied arrays.
    """
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()

IMAGE_SIZE = 64
CHANNELS = 1
BATCH_SIZE = 128
Z_DIM = 100
EPOCHS = 300
LOAD_MODEL = False
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999
LEARNING_RATE = 0.0002
NOISE_PARAM = 0.1

def load_data(data_path = "./data/lego-brick-images/dataset/"):
    train_data = utils.image_dataset_from_directory(
        data_path,
        labels=None,
        color_mode="grayscale",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        interpolation="bilinear",
    )

    def preprocess(img):
        """
        Normalize and reshape the images
        """
        img = (tf.cast(img, "float32") - 127.5) / 127.5
        return img


    train = train_data.map(lambda x: preprocess(x))

    return train

def build_discriminator():
    discriminator_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=False)(
        discriminator_input
    )
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(
        128, kernel_size=4, strides=2, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(
        256, kernel_size=4, strides=2, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(
        512, kernel_size=4, strides=2, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(
        1,
        kernel_size=4,
        strides=1,
        # no padding here, as the input here is 4x4 
        #  which is the same size as the kernel,
        #  the output is 1x1 now
        padding="valid",    
        use_bias=False,
        activation="sigmoid",
    )(x)
    # Dense layer can reduce the channel size to 1, but not the spatial size
    # x = layers.Dense(1, use_bias=False, activation="sigmoid")(x)
    discriminator_output = layers.Flatten()(x)

    discriminator = models.Model(discriminator_input, discriminator_output)
    return discriminator

discriminator = build_discriminator()
discriminator.summary()

def build_generator():
    generator_input = layers.Input(shape=(Z_DIM,))
    x = layers.Reshape((1, 1, Z_DIM))(generator_input)
    x = layers.Conv2DTranspose(
        512, kernel_size=4, strides=1, padding="valid", use_bias=False
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(
        256, kernel_size=4, strides=2, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(
        128, kernel_size=4, strides=2, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    # x = layers.Conv2DTranspose(
    #     64, kernel_size=4, strides=2, padding="same", use_bias=False
    # )(x)
    # replace with Upsampling
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(
        64, kernel_size=4, strides=1, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    generator_output = layers.Conv2DTranspose(
        CHANNELS,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False,
        activation="tanh",
    )(x)
    generator = models.Model(generator_input, generator_output)
    return generator

generator = build_generator()
generator.summary()