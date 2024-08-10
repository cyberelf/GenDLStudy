#!/usr/bin/env python
# coding: utf-8

# # 🥙 LSTM on Recipe Data

# In this notebook, we'll walk through the steps required to train your own LSTM on the recipes dataset


import numpy as np
import json
import re
import string
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, losses


import os
from gendlstudy.utils import sample_batch, display

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

VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 100
N_UNITS = 128
VALIDATION_SPLIT = 0.2
SEED = 42
LOAD_MODEL = False
BATCH_SIZE = 32
EPOCHS = 25


# ## 1. Load the data <a name="load"></a>

# Load the full dataset
with open(f"{data_dir}/full_format_recipes.json") as json_data:
    recipe_data = json.load(json_data)

# Filter the dataset
filtered_data = [
    "Recipe for " + x["title"] + " | " + " ".join(x["directions"])
    for x in recipe_data
    if "title" in x
    and x["title"] is not None
    and "directions" in x
    and x["directions"] is not None
]

# Count the recipes
n_recipes = len(filtered_data)
print(f"{n_recipes} recipes loaded")


example = filtered_data[9]
print(example)


# ## 2. Tokenise the data


# Pad the punctuation, to treat them as separate 'words'
def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s


text_data = [pad_punctuation(x) for x in filtered_data]


# Display an example of a recipe
example_data = text_data[9]
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

# ## 3. Create the Training Set

# Create the training set of recipes and the same text shifted by one word
def prepare_inputs(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y

train_ds = text_ds.map(prepare_inputs)

# Create a TextGenerator checkpoint
class TextGenerator(callbacks.Callback):
    def __init__(self, index_to_word, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }  # <1>

    def sample_from(self, probs, temperature):  # <2>
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]  # <3>
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:  # <4>
            x = np.array([start_tokens])
            y = self.model.predict(x, verbose=0)  # <5>
            sample_token, probs = self.sample_from(y[0][-1], temperature)  # <6>
            info.append({"prompt": start_prompt, "word_probs": probs})
            start_tokens.append(sample_token)  # <7>
            start_prompt = start_prompt + " " + self.index_to_word[sample_token]
        print(f"\ngenerated text:\n{start_prompt}\n")
        return info

    def on_epoch_end(self, epoch, logs=None):
        self.generate("recipe for", max_tokens=100, temperature=1.0)


# Create a model save checkpoint
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=f"{checkpoint_dir}/checkpoint.ckpt",
    save_weights_only=True,
    save_freq="epoch",
    verbose=0,
)

tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir)

# Tokenize starting prompt
text_generator = TextGenerator(vocab)

def train(continue_from=None):
    if continue_from:
        lstm = models.load_model(continue_from, compile=False)
    else:
        inputs = layers.Input(shape=(None,), dtype="int32")
        x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)
        x = layers.LSTM(N_UNITS, return_sequences=True)(x)
        outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)
        lstm = models.Model(inputs, outputs)
        lstm.summary()

    loss_fn = losses.SparseCategoricalCrossentropy()
    lstm.compile("adam", loss_fn)

    lstm.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback, tensorboard_callback, text_generator],
    )

    # Save the final model
    lstm.save(f"{model_dir}/lstm")

    return lstm

# lstm = train()
lstm = models.load_model(f"{model_dir}/lstm", compile=False)
text_generator.model = lstm

# ## 6. Generate text using the LSTM

def print_probs(info, vocab, top_k=5):
    for i in info:
        print(f"\nPROMPT: {i['prompt']}")
        word_probs = i["word_probs"]
        p_sorted = np.sort(word_probs)[::-1][:top_k]
        i_sorted = np.argsort(word_probs)[::-1][:top_k]
        for p, i in zip(p_sorted, i_sorted):
            print(f"{vocab[i]}:   \t{np.round(100*p,2)}%")
        print("--------\n")


info = text_generator.generate(
    "recipe for roasted vegetables | chop 1 /", max_tokens=10, temperature=1.0
)

print_probs(info, vocab)

info = text_generator.generate(
    "recipe for roasted vegetables | chop 1 /", max_tokens=100, temperature=0.2
)

print_probs(info, vocab)

info = text_generator.generate(
    "recipe for chocolate ice cream |", max_tokens=7, temperature=1.0
)
print_probs(info, vocab)


info = text_generator.generate(
    "recipe for chocolate ice cream |", max_tokens=7, temperature=0.2
)
print_probs(info, vocab)


