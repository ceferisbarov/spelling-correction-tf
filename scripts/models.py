import os
from datetime import datetime

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model

from load_data import (
    input_token_index,
    max_decoder_seq_length,
    max_encoder_seq_length,
    num_decoder_tokens,
    num_encoder_tokens,
    reverse_input_char_index,
    reverse_target_char_index,
    target_token_index,
)

latent_dim = 256  # Latent dimensionality of the encoding space.


class DeepEnsemble:
    @staticmethod
    def load_from_dir(directory, **kwargs):
        new = DeepEnsemble(**kwargs)
        models = []
        print(list(os.walk(directory)))
        for path in list(os.walk(directory))[0][1]:
            temp_model = []
            temp_model.append(load_model(os.path.join(directory, path, "training")))
            temp_model.append(load_model(os.path.join(directory, path, "encoder")))
            temp_model.append(load_model(os.path.join(directory, path, "decoder")))

            models.append(tuple(temp_model))

        new.models = models
        return new

    def __init__(self, no_models=5, threshold=0.8, name=None):
        self.no_models = no_models
        self.models = [self.Seq2SeqModel() for i in range(self.no_models)]
        self.threshold = threshold

        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = name if name else f"DeepEnsemble_{now}"

    def fit(self, **kwargs):
        if kwargs.get("verbose") is None:
            kwargs["verbose"] = 1
        for i in range(self.no_models):
            print(f"Training model no. {i+1}")
            self.models[i][0].fit(**kwargs)

    def predict(self, X):
        # Encode the sequence for the models
        input_seq = self.encode_for_inference(X)

        # No. of remaining models to run
        remaining = self.no_models

        # No. of matching predictions required
        k = self.threshold * self.no_models

        predictions = {}
        for i in self.models:
            out = self.decode_sequence(i[1], i[2], input_seq)
            predictions[out] = predictions.get(out, 0) + 1
            remaining -= 1

            if max(predictions.values()) + remaining < k:
                return X

            elif max(predictions.values()) >= k:
                return max(predictions, key=predictions.get)

        return X

    def save(self, save_dir=None):
        if save_dir == None:
            save_dir = self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        folder_names = [
            os.path.join(save_dir, f"model_{i+1}") for i in range(self.no_models)
        ]

        for i, path in enumerate(folder_names):
            if not os.path.exists(path):
                os.makedirs(path)

            self.models[i][0].save(os.path.join(path, f"training"))
            self.models[i][1].save(os.path.join(path, f"encoder"))
            self.models[i][2].save(os.path.join(path, f"decoder"))

    def encode_for_inference(self, input_text):
        encoder_input_text = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens), dtype="float32"
        )
        for t, char in enumerate(input_text):
            encoder_input_text[:, t, input_token_index[char]] = 1.0
        encoder_input_text[:, t + 1 :, input_token_index[" "]] = 1.0
        return encoder_input_text

    def Seq2SeqModel(self):
        # Define an input sequence and process it.
        encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
        encoder = keras.layers.LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = keras.layers.LSTM(
            latent_dim, return_sequences=True, return_state=True
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_inputs, initial_state=encoder_states
        )
        decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        encoder_inputs = model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_inputs = model.input[1]  # input_2
        decoder_state_input_h = keras.Input(shape=(latent_dim,))
        decoder_state_input_c = keras.Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

        return (model, encoder_model, decoder_model)

    def decode_sequence(self, encoder_model, decoder_model, input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index["\t"]] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]

        return decoded_sentence
