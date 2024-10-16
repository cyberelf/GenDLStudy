# # 🚀 GPT
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks
from keras.saving import register_keras_serializable


def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    # colunm vector 
    i = tf.range(n_dest)[:, None]
    # row vector
    j = tf.range(n_src)
    # lower triangular matrix
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)

# ## 6. Create a Transformer Block layer <a name="transformer"></a>
@register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attn = layers.MultiHeadAttention(
            num_heads, key_dim, output_shape=embed_dim
        )
        self.dropout_1 = layers.Dropout(self.dropout_rate)
        self.ln_1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = layers.Dense(self.ff_dim, activation="relu")
        self.ffn_2 = layers.Dense(self.embed_dim)
        self.dropout_2 = layers.Dropout(self.dropout_rate)
        self.ln_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(
            batch_size, seq_len, seq_len, tf.bool
        )
        attention_output, attention_scores = self.attn(
            inputs,
            inputs,
            attention_mask=causal_mask,
            return_attention_scores=True,
        )
        attention_output = self.dropout_1(attention_output)
        out1 = self.ln_1(inputs + attention_output)
        ffn_1 = self.ffn_1(out1)
        ffn_2 = self.ffn_2(ffn_1)
        ffn_output = self.dropout_2(ffn_2)
        return (self.ln_2(out1 + ffn_output), attention_scores)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


# ## 7. Create the Token and Position Embedding <a name="embedder"></a>
@register_keras_serializable()
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        # relpace the position embedding with the positional encoding
        # self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        # positional encoding
        self.positions = self.positional_encoding()

    def positional_encoding(self):
        positions = np.arange(self.max_len)[:, np.newaxis]
        i = np.arange(self.embed_dim)[np.newaxis, :]
        angle_rads = positions / 10000 ** ((2 * (i // 2)) / np.float32(self.embed_dim))
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)

    def call(self, x):
        # maxlen = tf.shape(x)[-1]
        # positions = tf.range(start=0, limit=maxlen, delta=1, dtype=tf.float32)
        # positions = self.pos_emb(positions)
        batch_size = tf.shape(x)[0]
        seqlen = tf.shape(x)[1]
        positions = self.positions[:, :seqlen, :]
        positions = tf.tile(positions, [batch_size, 1, 1])
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_len": self.max_len,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


def build_model(vocab_size, max_len, embed_dim, key_dim, num_heads, ff_dim, num_transformer_blocks, dropout_rate=0.1) -> models.Model:
    inputs = layers.Input(shape=(None,), dtype=tf.int32)
    x = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)(inputs)
    x, attention_scores = TransformerBlock(
        num_heads, key_dim, embed_dim, ff_dim, dropout_rate=dropout_rate
    )(x)
    for i in range(num_transformer_blocks - 1):
        x, _ = TransformerBlock(
            num_heads, key_dim, embed_dim, ff_dim
        )(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    gpt = models.Model(inputs=inputs, outputs=[outputs, attention_scores])
    gpt.compile("adam", loss=[losses.SparseCategoricalCrossentropy(), None])
    return gpt

def load_model_from_file(model_file):
    """Load a model from a file."""
    return models.load_model(model_file)

# Create a TextGenerator checkpoint
class TextGenerator(callbacks.Callback):
    def __init__(self, index_to_word, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }

    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y, att = self.model.predict(x, verbose=0)
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            info.append(
                {
                    "prompt": start_prompt,
                    "word_probs": probs,
                    "atts": att[0, :, -1, :],
                }
            )
            start_tokens.append(sample_token)
            start_prompt = start_prompt + " " + self.index_to_word[sample_token]
        print(f"\ngenerated text:\n{start_prompt}\n")
        return info

    def on_epoch_end(self, epoch, logs=None):
        self.generate("wine review", max_tokens=80, temperature=1.0)



def print_probs(info, vocab, top_k=5):
    colors = ['\033[0;36m', '\033[0;33m', '\033[0;31m']
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

