import numpy as np
from tensorflow import keras
import tensorflow as tf

from load_data import (
    input_token_index,
    max_decoder_seq_length,
    max_encoder_seq_length,
    num_decoder_tokens,
    num_encoder_tokens,
    reverse_target_char_index,
    target_token_index,
)

latent_dim = 256  # Latent dimensionality of the encoding space.


class Model:
    """A class for Seq2Seq ensemble models"""

    @staticmethod
    def load_from_dir(directory, **kwargs):
        pass

    def _quantize_model(self, model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        ]

        tflite_model = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        return interpreter

    def fit(self, **kwargs):
        pass

    def predict(self, x, threshold=None, certain=False):
        pass

    def save(self, save_dir=None):
        pass

    def quantize(self, include_full_model=False):
        pass

    def encode_for_inference(self, input_text):
        encoder_input_text = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens), dtype="float32"
        )
        for t, char in enumerate(input_text):
            encoder_input_text[:, t, input_token_index[char]] = 1.0
        encoder_input_text[:, t + 1 :, input_token_index[" "]] = 1.0
        return encoder_input_text

    def seq2seq_model(self):
        # Define an input sequence and process it.
        encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
        encoder = keras.layers.LSTM(latent_dim, return_state=True)
        _, state_h, state_c = encoder(encoder_inputs)

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
        _, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
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
        input_kws = encoder_model._get_full_signature_list()["serving_default"][
            "inputs"
        ].keys()
        output_kws = list(
            encoder_model._get_full_signature_list()["serving_default"][
                "outputs"
            ].keys()
        )
        inputs = {i: j for i, j in zip(input_kws, [input_seq])}
        encode = encoder_model.get_signature_runner("serving_default")
        encoded = encode(**inputs)
        s_1 = encoded[output_kws[0]]
        s_2 = encoded[output_kws[1]]

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens), dtype="float32")
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index["\t"]] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        decoded_vectors = []
        while not stop_condition:
            input_kws = decoder_model._get_full_signature_list()["serving_default"][
                "inputs"
            ].keys()
            output_kws = list(
                decoder_model._get_full_signature_list()["serving_default"][
                    "outputs"
                ].keys()
            )
            inputs = {i: j for i, j in zip(input_kws, [target_seq, s_1, s_2])}
            decode = decoder_model.get_signature_runner("serving_default")
            decoded = decode(**inputs)

            output_tokens = decoded[output_kws[0]]
            c = decoded[output_kws[1]]
            h = decoded[output_kws[2]]

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char
            decoded_vectors.append(output_tokens)

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens), dtype="float32")
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            s_1, s_2 = c, h

        return decoded_sentence, decoded_vectors
