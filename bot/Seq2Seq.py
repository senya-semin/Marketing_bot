import os
from google.protobuf import message
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D, Add
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
import numpy as np
from Preparator import Preparator
from Pupsen import Pupsen
import tensorflow_addons as tfa
from Encoder_Decoder import Encoder, Decoder
import re


class Seq2Seq(Preparator):
    def __init__(self) -> None:
        self.lengh = 65
        self.embeding_dims = 128
        self.batch_size = 1270
        self.rnn_units = 1024
        self.epochs = 150
        #
        self.library = []

        self.alphabet = Preparator.alphabet
        self.strat_token = Preparator.strat_token
        self.end_token = Preparator.end_token
        self.fill_token = Preparator.fill_token

        self.encoder = Encoder(43, self.embeding_dims, self.rnn_units)
        self.decoder = Decoder(43, self.embeding_dims, self.rnn_units, self.batch_size)
        self.optimizer = tf.keras.optimizers.Adam()

        self.checkpointdir = os.path.join("chekpoint", "Vupsen_model")
        self.chkpoint_prefix = os.path.join(self.checkpointdir, "chkpoint")
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            encoderNetwork=self.encoder,
            decoderNetwork=self.decoder,
        )

    def save_model(self):
        self.checkpoint.save(file_prefix=self.chkpoint_prefix)
        return

    def load_model(self):
        return tf.train.load_variable(
            self.checkpointdir,
            "decoderNetwork/decoder_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE",
        )

    def loss_function(self, y_pred, y):
        sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

        loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
        mask = tf.logical_not(tf.math.equal(y, 0))  # output 0 for y=0 else output 1
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = mask * loss
        loss = tf.reduce_mean(loss)
        return loss

    def initialize_initial_state(self, batch_size=10):
        return [
            tf.zeros((batch_size, self.rnn_units)),
            tf.zeros((batch_size, self.rnn_units)),
        ]

    def train(self, input, output, encoder_initial_cell_state, batch_size):
        loss = 0
        with tf.GradientTape() as tape:
            encoder_emb_inp = self.encoder.encoder_embedding(input)
            a, a_tx, c_tx = self.encoder.encoder_rnnlayer(
                encoder_emb_inp, initial_state=encoder_initial_cell_state
            )

            # Prepare correct Decoder input & output sequence data
            decoder_input = output[:, :-1]  # ignore <end>
            # compare logits with timestepped +1 version of decoder_input
            decoder_output = output[:, 1:]  # ignore <start>
            # Decoder Embeddings
            decoder_emb_inp = self.decoder.decoder_embedding(decoder_input)
            self.decoder.attention_mechanism.setup_memory(a)
            decoder_initial_state = self.decoder.build_decoder_initial_state(
                batch_size, encoder_state=[a_tx, c_tx], Dtype=tf.float32
            )
            outputs, _, _ = self.decoder.decoder(
                decoder_emb_inp,
                initial_state=decoder_initial_state,
                sequence_length=batch_size * [self.lengh - 1],
            )
            logits = outputs.rnn_output

            loss = self.loss_function(logits, decoder_output)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)

        # grads_and_vars – List of(gradient, variable) pairs.
        grads_and_vars = zip(gradients, variables)
        self.optimizer.apply_gradients(grads_and_vars)
        return loss

    def learn(self):
        data = pd.read_csv("общение.csv", delimiter=";")
        text, answer = self.get_book(self.library)
        text += data["respond"].to_list()
        answer += data["answer"].to_list()
        print(answer[-1])
        lenght = []
        for i in range(len(answer)):
            lenght += [len(answer[i])]

        print(np.mean(lenght))
        print(np.median(lenght))
        print(np.max(lenght))
        print(len(answer))

        batch_count = len(answer)
        print(
            "------------------------------------------------------------------------\n",
            "start\n",
            "------------------------------------------------------------------------\n",
        )
        
        input = [self.alphabet_text(sentence, lengh=self.lengh) for sentence in text]
        output = [self.alphabet_text(sentence, lengh=self.lengh) for sentence in answer]

        for i in range(self.epochs):

            batch = 0
            step = 10
            total_loss = 0.0

            while step < batch_count:
                if step > batch_count:
                    step = batch_count
                batch_size = step - batch
                encoder_initial_cell_state = self.initialize_initial_state(batch_size)
                input_batch = np.array(input[batch:step])
                output_batch = np.array(output[batch:step])
                input_batch = tf.convert_to_tensor(input_batch)
                output_batch = tf.convert_to_tensor(output_batch)
                batch_loss = self.train(
                    input_batch, output_batch, encoder_initial_cell_state, batch_size
                )
                total_loss += batch_loss
                batch += self.batch_size
                step += self.batch_size
                print(
                    "total loss: {} epoch {} batch {} / {} ".format(
                        batch_loss.numpy(), i + 1, batch, batch_count
                    )
                )
        self.save_model()
        self.encoder.save_weights("weights/weight_encoder")
        self.decoder.save_weights("weights/weight_decoder")
        print(
            "\n"
            "------------------------------------------------------------------------\n",
            "finish\n",
            "------------------------------------------------------------------------\n",
        )

    def predict(self, input_text):
        self.encoder.load_weights("weights/weight_encoder")
        self.decoder.load_weights("weights/weight_decoder")

        input_text = self.alphabet_text(input_text, lengh=self.lengh)

        decoder_embedding_matrix = self.load_model()

        encoder_initial_cell_state = self.initialize_initial_state(1)
        encoder_emb_inp = self.encoder.encoder_embedding(input_text)
        print(len(encoder_emb_inp))
        encoder_emb_inp = np.reshape(
            encoder_emb_inp, (1, self.lengh, self.embeding_dims)
        )

        a, a_tx, c_tx = self.encoder.encoder_rnnlayer(
            encoder_emb_inp, initial_state=encoder_initial_cell_state
        )
        start_tokens = tf.fill(1, 41)
        end_token = 42
        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        decoder_input = input_text[:-1]

        decoder_emb_inp = self.decoder.decoder_embedding(decoder_input)

        decoder_instance = tfa.seq2seq.BasicDecoder(
            cell=self.decoder.rnn_cell,
            sampler=greedy_sampler,
            output_layer=self.decoder.dense_layer,
        )
        self.decoder.attention_mechanism.setup_memory(a)

        # print(decoder_instance)
        decoder_initial_state = self.decoder.build_decoder_initial_state(
            1, encoder_state=[a_tx, c_tx], Dtype=tf.float32
        )
        print("sdtep")
        print(np.array(decoder_initial_state).shape)
        (first_finished, first_inputs, first_state) = decoder_instance.initialize(
            decoder_embedding_matrix,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state,
        )

        inputs = first_inputs
        state = first_state
        predictions = np.empty((300, 0), dtype=np.int32)
        for i in range(self.lengh):
            outputs, next_state, next_inputs, finished = decoder_instance.step(
                i, inputs, state
            )
            inputs = next_inputs
            state = next_state
            outputs = np.expand_dims(outputs.sample_id, axis=-1)
            predictions = np.append(predictions, outputs)
        print(self.to_alphabet(predictions))
        return self.to_alphabet(predictions)


