import tensorflow as tf
import numpy as np
import time, re

from dataset_utils import preprocess_sentence


def gumbel_softmax(o_t, temperature, eps=1e-20):
  """ Sample from Gumbel(0, 1) """
  u = tf.random.uniform(tf.shape(o_t),minval=0, maxval=1)
  g_t = -tf.math.log(-tf.math.log(u + eps) + eps)

  gumbel_t = tf.math.add(o_t, g_t)
  return tf.math.multiply(gumbel_t, temperature)


# ========================================================================================
# Sequence to Sequence
# Source: https://www.tensorflow.org/text/tutorials/nmt_with_attention
# ========================================================================================
class Seq2Seq:
    def __init__(self, inp_lang, targ_lang, max_length, vocab_inp_size, vocab_tar_size,
                 batch_size=64, embedding_dim=256, units= 1024, loss_object=None, optimizer=None,
                 w_atten=False, gumbel=False, gumbel_temp=0.5, bpe=False):
        self.BATCH_SIZE = batch_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.w_atten = w_atten
        self.gumbel = gumbel
        self.gumbel_temp = gumbel_temp
        self.bpe = bpe

        self.inp_lang = inp_lang
        self.targ_lang = targ_lang
        self.max_length_inp, self.max_length_targ = max_length

        self.vocab_inp_size = vocab_inp_size
        self.vocab_tar_size = vocab_tar_size

        if self.bpe:
            self.targ_lang_start_idx = self.targ_lang.tokenize("<start>")
            self.targ_lang_start_idx = self.targ_lang_start_idx.numpy()[0]
            self.targ_lang_end_idx   = self.targ_lang.tokenize("<end>")
            self.targ_lang_end_idx   = tf.cast(self.targ_lang_end_idx, tf.int32)
        else:
            self.targ_lang_start_idx = self.targ_lang.word_index['<start>']

        if loss_object is not None: self.loss_object = loss_object
        else: self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if optimizer is not None: self.optimizer = optimizer
        else: self.optimizer = tf.keras.optimizers.RMSprop()

        self.encoder = Seq2Seq.Encoder(self.vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)
        self.decoder = Seq2Seq.Decoder(self.vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE, self.w_atten, self.gumbel, self.gumbel_temp)


    def loss_function(self, real, pred):
      mask = tf.math.logical_not(tf.math.equal(real, 0))
      loss_ = self.loss_object(real, pred)
      mask = tf.cast(mask, dtype=loss_.dtype)
      loss_ *= mask
      return tf.reduce_mean(loss_)


    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.targ_lang_start_idx] * self.BATCH_SIZE, 1)

            for t in range(1, targ.shape[1]):
                if self.w_atten:
                    predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                else:
                    predictions, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output)
                loss += self.loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss


    def train(self, dataset, steps_per_epoch, epochs):
      for epoch in range(epochs):
          start = time.time()

          enc_hidden = self.encoder.initialize_hidden_state()
          total_loss = 0
          for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = self.train_step(inp, targ, enc_hidden)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
          print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
          print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


    def evaluate(self, sentence):
        # attention_plot = np.zeros((max_length_targ, max_length_inp))
        sentence = preprocess_sentence(sentence)

        if self.bpe:
            inputs = [self.inp_lang.tokenize(i) if i != 0
                                                else self.inp_lang.tokenize('<unkown>') for i in sentence.split(' ')]
            inputs = tf.concat(inputs, 0)
            enc_input = tf.expand_dims(inputs, 0)

            dec_input = self.targ_lang.tokenize('<start>')
            # dec_input = tf.concat(dec_input, 0)
            dec_input = tf.expand_dims(dec_input, 0)

        else:
            inputs = [self.inp_lang.word_index[i] if i in self.inp_lang.word_index
                                                  else self.inp_lang.word_index['<unkown>'] for i in sentence.split(' ')]
            inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=self.max_length_inp, padding='post')
            enc_input = tf.convert_to_tensor(inputs)

            dec_input = tf.expand_dims([self.targ_lang.word_index['<start>']], 0)

        result = ''
        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(enc_input, hidden)
        dec_hidden = enc_hidden


        for t in range(self.max_length_targ):
            if self.w_atten:
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_out)
                predicted_id = tf.argmax(predictions[0]).numpy()
            else:
                predictions, dec_hidden = self.decoder(dec_input, dec_hidden, enc_out)
                predicted_id = tf.argmax(predictions[0][0]).numpy()

            if predicted_id != 0:
                if self.bpe:
                    predicted_word = self.targ_lang.detokenize([[predicted_id]])
                    predicted_word = predicted_word.numpy()[0, 0].decode('utf-8')
                    result += predicted_word + ' '
                else:
                    result += self.targ_lang.index_word[predicted_id] + ' '

                if ((self.bpe and tf.equal(predicted_id, self.targ_lang_end_idx)) or
                    (not self.bpe and self.targ_lang.index_word[predicted_id] == '<end>')):
                    return result, sentence

            dec_input = tf.expand_dims([predicted_id], 0)

        result = re.sub(" ##", '', result) if self.bpe else result

        return result, sentence


    class Encoder(tf.keras.Model):
      def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Seq2Seq.Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

      def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

      def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


    class Decoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, w_atten=False, gumbel=False, gumbel_temp=0.5):
            super(Seq2Seq.Decoder, self).__init__()
            self.batch_sz = batch_sz
            self.dec_units = dec_units
            self.w_atten = w_atten
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(self.dec_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')

            if gumbel: self.fc = tf.keras.layers.Dense(vocab_size, activation=(lambda x: gumbel_softmax(x, temperature=gumbel_temp)))
            else: self.fc = tf.keras.layers.Dense(vocab_size)


            if self.w_atten: self.attention = Seq2Seq.BahdanauAttention(self.dec_units)

        def call(self, x, hidden, enc_output):
            if self.w_atten:
                context_vector, attention_weights = self.attention(hidden, enc_output)
                x = self.embedding(x)
                x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
                output, state = self.gru(x)
                output = tf.reshape(output, (-1, output.shape[2]))
                x = self.fc(output)
                return x, state, attention_weights
            else:
                x = self.embedding(x)
                output, state = self.gru(x, hidden)
                x = self.fc(output)
                return x, state


    class BahdanauAttention(tf.keras.Model):
      def __init__(self, units):
        super(Seq2Seq.BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

      def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights



# ========================================================================================
# Transformer
# Source: https://www.tensorflow.org/text/tutorials/transformer
# ========================================================================================
class Transformer(tf.keras.Model):
    def __init__(self, inp_lang, targ_lang, max_length, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, rate=0.1, loss_object=None, optimizer=None, gumbel=False):

        super(Transformer, self).__init__()
        self.inp_lang = inp_lang
        self.targ_lang = targ_lang
        self.max_length = max_length

        self.encoder = Transformer.Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
        self.decoder = Transformer.Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)
        if gumbel: self.final_layer = tf.keras.layers.Dense(target_vocab_size, activation=(lambda x: gumbel_softmax(x, temperature=0.5)))
        else: self.final_layer = tf.keras.layers.Dense(target_vocab_size)

        if loss_object is not None: self.loss_object = loss_object
        else: self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        if optimizer is not None: self.optimizer = optimizer
        else: self.optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights


    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)


    @tf.function
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self(inp, tar_inp,
                                         True,
                                         enc_padding_mask, combined_mask, dec_padding_mask)
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)


    def train(self, epochs, dataset):
        for epoch in range(epochs):
            start = time.time()
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, (inp, tar)) in enumerate(dataset):
                self.train_step(inp, tar)

                if batch % 500 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                                  batch,
                                                                                  self.train_loss.result(),
                                                                                  self.train_accuracy.result()))

            print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                 self.train_loss.result(),
                                                                 self.train_accuracy.result()))
            print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


    def evaluate(self, inp_sentence):
        sentence = preprocess_sentence(inp_sentence)
        inputs = [self.inp_lang.word_index[i] if i in self.inp_lang.word_index
                                              else self.inp_lang.word_index['<unkown>'] for i in sentence.split(' ')]
        encoder_input = tf.expand_dims(inputs, 0)
        decoder_input = [self.targ_lang.word_index['<start>']]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(self.max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = transformer.create_masks(encoder_input, output)

            predictions, attention_weights = self(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
            predictions = predictions[: ,-1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            if tf.equal(predicted_id, self.targ_lang.word_index['<end>']):
                output = tf.squeeze(output, axis=0)
                predicted_sentence = ' '.join([self.targ_lang.index_word[i] for i in output.numpy() if i != 0])
                return predicted_sentence, attention_weights

            output = tf.concat([output, predicted_id], axis=-1)

        output = tf.squeeze(output, axis=0)
        predicted_sentence = ' '.join([self.targ_lang.index_word[i] for i in output.numpy() if i != 0])
        return predicted_sentence, attention_weights


    def positional_encoding(position, d_model):
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
            return pos * angle_rates

        angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)


    def create_masks(self, inp, tar):
        def create_padding_mask(seq):
            seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
            return seq[:, tf.newaxis, tf.newaxis, :]

        def create_look_ahead_mask(size):
            mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
            return mask

        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = create_padding_mask(inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask


    def scaled_dot_product_attention(q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights


    def point_wise_feed_forward_network(d_model, dff):
        return tf.keras.Sequential([ tf.keras.layers.Dense(dff, activation='relu'),
                                     tf.keras.layers.Dense(d_model)
                                    ])


    class MultiHeadAttention(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads):
            super(Transformer.MultiHeadAttention, self).__init__()
            self.num_heads = num_heads
            self.d_model = d_model

            assert d_model % self.num_heads == 0

            self.depth = d_model // self.num_heads

            self.wq = tf.keras.layers.Dense(d_model)
            self.wk = tf.keras.layers.Dense(d_model)
            self.wv = tf.keras.layers.Dense(d_model)
            self.dense = tf.keras.layers.Dense(d_model)

        def split_heads(self, x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def call(self, v, k, q, mask):
            batch_size = tf.shape(q)[0]

            q = self.wq(q)  # (batch_size, seq_len, d_model)
            k = self.wk(k)  # (batch_size, seq_len, d_model)
            v = self.wv(v)  # (batch_size, seq_len, d_model)

            q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
            k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
            v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

            # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
            # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
            scaled_attention, attention_weights = Transformer.scaled_dot_product_attention(
            q, k, v, mask)

            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

            concat_attention = tf.reshape(scaled_attention,
            (batch_size, -1, self.d_model))  # (batch_size, seq_len_v, d_model)

            output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)
            return output, attention_weights


    class LayerNormalization(tf.keras.layers.Layer):
        def __init__(self, epsilon=1e-6, **kwargs):
            self.epsilon = epsilon
            super(Transformer.LayerNormalization, self).__init__(**kwargs)

        def build(self, input_shape):
            self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                         initializer= tf.keras.initializers.Ones(), trainable=True)
            self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                         initializer= tf.keras.initializers.Zeros(), trainable=True)
            super(Transformer.LayerNormalization, self).build(input_shape)

        def call(self, x):
            mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
            std = tf.keras.backend.std(x, axis=-1, keepdims=True)
            return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

        def compute_output_shape(self, input_shape):
            return input_shape


    class EncoderLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, rate=0.1):
            super(Transformer.EncoderLayer, self).__init__()

            self.mha = Transformer.MultiHeadAttention(d_model, num_heads)
            self.ffn = Transformer.point_wise_feed_forward_network(d_model, dff)

            self.layernorm1 = Transformer.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = Transformer.LayerNormalization(epsilon=1e-6)

            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)

        def call(self, x, training, mask):
            attn_output, _ = self.mha(x, x, x, mask)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(x + attn_output)

            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = self.layernorm2(out1 + ffn_output)
            return out2


    class DecoderLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, rate=0.1):
            super(Transformer.DecoderLayer, self).__init__()

            self.mha1 = Transformer.MultiHeadAttention(d_model, num_heads)
            self.mha2 = Transformer.MultiHeadAttention(d_model, num_heads)

            self.ffn = Transformer.point_wise_feed_forward_network(d_model, dff)

            self.layernorm1 = Transformer.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = Transformer.LayerNormalization(epsilon=1e-6)
            self.layernorm3 = Transformer.LayerNormalization(epsilon=1e-6)

            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)
            self.dropout3 = tf.keras.layers.Dropout(rate)


        def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
            attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
            attn1 = self.dropout1(attn1, training=training)
            out1 = self.layernorm1(attn1 + x)

            attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
            attn2 = self.dropout2(attn2, training=training)
            out2 = self.layernorm2(attn2 + out1)

            ffn_output = self.ffn(out2)
            ffn_output = self.dropout3(ffn_output, training=training)
            out3 = self.layernorm3(ffn_output + out2)

            return out3, attn_weights_block1, attn_weights_block2


    class Encoder(tf.keras.layers.Layer):
        def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
            super(Transformer.Encoder, self).__init__()
            self.d_model = d_model
            self.num_layers = num_layers

            self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
            self.pos_encoding = Transformer.positional_encoding(input_vocab_size, self.d_model)

            self.enc_layers = [Transformer.EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

            self.dropout = tf.keras.layers.Dropout(rate)

        def call(self, x, training, mask):
            seq_len = tf.shape(x)[1]
            x = self.embedding(x)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]
            x = self.dropout(x, training=training)
            for i in range(self.num_layers):
                x = self.enc_layers[i](x, training, mask)
            return x

    class Decoder(tf.keras.layers.Layer):
        def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
            super(Transformer.Decoder, self).__init__()

            self.d_model = d_model
            self.num_layers = num_layers

            self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
            self.pos_encoding = Transformer.positional_encoding(target_vocab_size, self.d_model)

            self.dec_layers = [Transformer.DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
            self.dropout = tf.keras.layers.Dropout(rate)

        def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
            seq_len = tf.shape(x)[1]
            attention_weights = {}
            x = self.embedding(x)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]
            x = self.dropout(x, training=training)

            for i in range(self.num_layers):
                x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
                attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
                attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

            return x, attention_weights
