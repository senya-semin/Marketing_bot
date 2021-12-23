from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow_addons.utils.types import Activation
from Preparator import Preparator
import tensorflow_addons as tfa

# source: https://github.com/dhirensk/ai/blob/master/English_to_French_seq2seq_tf_2_0_withAttention.ipynb

# ENCODER
class Encoder(tf.keras.Model):
    def __init__(self, input_vocab_size, embedding_dims, rnn_units):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(
            input_dim=input_vocab_size, output_dim=embedding_dims
        )
        self.encoder_rnnlayer = tf.keras.layers.LSTM(
            rnn_units, return_sequences=True, return_state=True
        )


# DECODER
class Decoder(tf.keras.Model):
    def __init__(self, output_vocab_size, embedding_dims, rnn_units, batch_size):
        super().__init__()
        self.rnn_units = rnn_units
        self.batch_size = batch_size
        self.decoder_embedding = tf.keras.layers.Embedding(
            input_dim=output_vocab_size, output_dim=embedding_dims
        )
        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(
            rnn_units, None, self.batch_size 
        )
        self.rnn_cell = self.build_rnn_cell(self.batch_size)
        self.decoder = tfa.seq2seq.BasicDecoder(
            self.rnn_cell, sampler=self.sampler, output_layer=self.dense_layer
        )

    def build_attention_mechanism(self, units, memory, memory_sequence_length):
        return tfa.seq2seq.LuongAttention(
            units, memory=memory, memory_sequence_length=memory_sequence_length
        )
        # return tfa.seq2seq.BahdanauAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

    # wrap decodernn cell
    def build_rnn_cell(self, batch_size):
        rnn_cell = tfa.seq2seq.AttentionWrapper(
            self.decoder_rnncell,
            self.attention_mechanism,
            attention_layer_size= self.rnn_units,
        )
        return rnn_cell

    def build_decoder_initial_state(self, batch_size, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=batch_size, dtype=Dtype
        )
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state
