#!/usr/bin/env python
# coding: utf-8

# # 🚀 GPT

# In this notebook, we'll walk through the steps required to train your own GPT model on the wine review dataset

# The code is adapted from the excellent [GPT tutorial](https://keras.io/examples/generative/text_generation_with_miniature_gpt/) created by Apoorv Nandan available on the Keras website.

import os
import numpy as np
import json
import re
import string

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from gendlstudy.gpt.model import TextGenerator, build_model, load_model_from_file


# ## 0. Parameters <a name="parameters"></a>

VOCAB_SIZE = 10000
MAX_LEN = 80
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 2
FEED_FORWARD_DIM = 256
VALIDATION_SPLIT = 0.2
SEED = 42
LOAD_MODEL = True
BATCH_SIZE = 32
EPOCHS = 5
TF_BLOCKS = 4

# data_dir = "/home/guangyu/workspace/dataset/wine-reviews"
data_dir = "/gemini/data-2/wine-review"
base_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = f"{base_dir}/checkpoint"
output_dir = f"{base_dir}/output"
log_dir = f"{base_dir}/logs"
model_dir = f"{base_dir}/models"

for directory in [checkpoint_dir, output_dir, log_dir, model_dir]:
    os.makedirs(directory, exist_ok=True)


model_file = f"{model_dir}/gpt_4tb.keras"
# ## 1. Load the data <a name="load"></a>

# Load the full dataset
with open(f"{data_dir}/winemag-data-130k-v2.json") as json_data:
    wine_data = json.load(json_data)


wine_data[10]


# Filter the dataset
filtered_data = [
    "wine review : "
    + x["country"]
    + " : "
    + x["province"]
    + " : "
    + x["variety"]
    + " : "
    + x["description"]
    for x in wine_data
    if x["country"] is not None
    and x["province"] is not None
    and x["variety"] is not None
    and x["description"] is not None
]


# Count the recipes
n_wines = len(filtered_data)
print(f"{n_wines} recipes loaded")


example = filtered_data[25]
print(example)


# ## 2. Tokenize the data <a name="tokenize"></a>

# Pad the punctuation, to treat them as separate 'words'
def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}, '\n'])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s


text_data = [pad_punctuation(x) for x in filtered_data]


# Display an example of a recipe
example_data = text_data[25]
example_data


# Convert to a Tensorflow Dataset
text_ds = (
    tf.data.Dataset.from_tensor_slices(text_data)
    .batch(BATCH_SIZE)
    .shuffle(1000)
)


# Create a vectorisation layer
vectorize_layer = layers.TextVectorization(
    standardize="lower",
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN + 1,
)


# Adapt the layer to the training set
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()


# Display some token:word mappings
for i, word in enumerate(vocab[:10]):
    print(f"{i}: {word}")


# Display the same example converted to ints
example_tokenised = vectorize_layer(example_data)
print(example_tokenised.numpy())


# ## 3. Create the Training Set <a name="create"></a>

# Create the training set of recipes and the same text shifted by one word
def prepare_inputs(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


train_ds = text_ds.map(prepare_inputs)


# ## 9. Train the Transformer <a name="train"></a>

# Tokenize starting prompt
text_generator = TextGenerator(vocab)

if LOAD_MODEL:
    # model.load_weights('./models/model')
    gpt = load_model_from_file(model_file)
    gpt.summary()
    text_generator.model = gpt
else:
    gpt = build_model(
        VOCAB_SIZE,
        MAX_LEN,
        EMBEDDING_DIM,
        KEY_DIM,
        N_HEADS,
        FEED_FORWARD_DIM,
        TF_BLOCKS,
    )
    gpt.summary()
    # Create a model save checkpoint
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=f"{checkpoint_dir}/checkpoint.weights.h5",
        save_weights_only=True,
        save_freq="epoch",
        verbose=0,
    )

    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir)

    gpt.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback, tensorboard_callback, text_generator],
    )


    # Save the final model
    gpt.save(model_file)


# # 3. Generate text using the Transformer
colors = ['\033[0;36m', '\033[0;33m', '\033[0;31m']
def print_probs(info, vocab, top_k=5):
    for i in info:
        highlighted_text = []
        for word, att_score in zip(
            i["prompt"].split(), np.mean(i["atts"], axis=0)
        ):
            color_id = int(att_score * 2 // max(np.mean(i["atts"], axis=0)))
            highlighted_text.append(
                str(colors[color_id])
                + word
                + "\033[0m"
            )
        highlighted_text = " ".join(highlighted_text)
        print("\t" + highlighted_text)

        word_probs = i["word_probs"]
        p_sorted = np.sort(word_probs)[::-1][:top_k]
        i_sorted = np.argsort(word_probs)[::-1][:top_k]
        for p, i in zip(p_sorted, i_sorted):
            print(f"{vocab[i]}:   \t{np.round(100*p,2)}%")
        print("--------\n")


info = text_generator.generate(
    "wine review : us", max_tokens=80, temperature=1.0
)


info = text_generator.generate(
    "wine review : italy", max_tokens=80, temperature=0.5
)


info = text_generator.generate(
    "wine review : germany", max_tokens=80, temperature=0.5
)
print_probs(info, vocab)

