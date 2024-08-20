import tensorflow as tf

class LSTMCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LSTMCell, self).__init__()
        self.units = units
        self.state_size = (units, units)

        # Define weights for input gate, forget gate, output gate, and cell state
        self.W_f = tf.keras.layers.Dense(units, use_bias=True)
        self.W_i = tf.keras.layers.Dense(units, use_bias=True)
        self.W_o = tf.keras.layers.Dense(units, use_bias=True)
        self.W_c = tf.keras.layers.Dense(units, use_bias=True)

    def call(self, x, states):
        h_prev, c_prev = states

        # Forget gate
        f_t = tf.sigmoid(self.W_f(tf.concat([x, h_prev], axis=-1)))

        # Input gate
        i_t = tf.sigmoid(self.W_i(tf.concat([x, h_prev], axis=-1)))
        
        # Candidate cell state
        c_tilde = tf.tanh(self.W_c(tf.concat([x, h_prev], axis=-1)))

        # Cell state
        c_t = f_t * c_prev + i_t * c_tilde

        # Output gate
        o_t = tf.sigmoid(self.W_o(tf.concat([x, h_prev], axis=-1)))

        # Hidden state
        h_t = o_t * tf.tanh(c_t)

        return h_t, [h_t, c_t]

class LSTMModel(tf.keras.Model):
    def __init__(self, units):
        super(LSTMModel, self).__init__()
        self.units = units
        self.lstm_cell = LSTMCell(units)
        self.rnn = tf.keras.layers.RNN(self.lstm_cell, unroll=True, return_sequences=True, return_state=True)
        # self.dense = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        h0 = tf.zeros((batch_size, self.units))
        c0 = tf.zeros((batch_size, self.units))
        initial_state = [h0, c0]
        output, final_memory_state, final_carry_state = self.rnn(inputs, initial_state=initial_state)
        # output = self.dense(output)
        return output

