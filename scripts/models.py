import os
from datetime import datetime

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow as tf

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
from utils import shuffle_matrices_by_row

latent_dim = 256  # Latent dimensionality of the encoding space.

class Model:
    """A class for Seq2Seq ensemble models"""

    @staticmethod
    def load_from_dir(directory, **kwargs):
        """
        Static method to load model from the given directory.
        The model should be saved in SaveModel format
        """
        new = Model(**kwargs)
        model = []
        model.append(
            keras.models.load_model(os.path.join(directory, path, "training"))
        )
        model.append(
            keras.models.load_model(os.path.join(directory, path, "encoder"))
        )
        model.append(
            keras.models.load_model(os.path.join(directory, path, "decoder"))
        )

        new.model = model

        return new

    def __init__(
        self,
        threshold=0.95,
        name=None,
    ):
        self.model = self.seq2seq_model()
        self.threshold = threshold

        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = name if name else f"Singleton_{now}"

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
        if kwargs.get("verbose") is None:
            kwargs["verbose"] = 1

        encoder_input_data, decoder_input_data = kwargs.get("x")[0], kwargs.get("x")[1]
        decoder_target_data = kwargs.get("y")

        encoder_input_data, decoder_input_data, decoder_target_data = shuffle_matrices_by_row(encoder_input_data, decoder_input_data, decoder_target_data)

        kwargs["x"] = [encoder_input_data, decoder_input_data]
        kwargs["y"] = decoder_target_data

        self.models[0].fit(**kwargs)

    def predict(self, x, threshold=None, certain=False):
        if not threshold:
            threshold = self.threshold
            
        model = self.models[0]

        input_seq = self.encode_for_inference(x)

        word, vectors = self.decode_sequence(model[1], model[2], input_seq)
        delta = 0
        for vector in vectors:
            temp = vector.flatten()
            temp = np.sort(temp)
            delta += temp[-1].item() - temp[-2].item()

        delta = delta / (len(vectors))
        if certain:
            return word if delta >= threshold else x
        else:
            return word
    
    def save(self, save_dir=None):
        if save_dir is None:
            save_dir = self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.model[0].save(os.path.join(save_dir, "training"))
        self.model[1].save(os.path.join(save_dir, "encoder"))
        self.model[2].save(os.path.join(save_dir, "decoder"))

    def quantize(self, include_full_model=False):
        self.models = [list(i) for i in self.models]
        if include_full_model:
            model_range = range(0, 3)
        else:
            model_range = range(1, 3)

        for j in model_range:
            self.models[i][j] = self._quantize_model(self.models[i][j])

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

class DeepEnsemble(Model):
    """DeepEnsemble class for Seq2Seq ensemble models"""

    @staticmethod
    def load_from_dir(directory, **kwargs):
        """
        Static method to load model from the given directory.
        The model should be saved in SaveModel format
        """
        new = DeepEnsemble(**kwargs)
        models = []
        for path in list(os.walk(directory))[0][1]:
            temp_model = []
            temp_model.append(
                keras.models.load_model(os.path.join(directory, path, "training"))
            )
            temp_model.append(
                keras.models.load_model(os.path.join(directory, path, "encoder"))
            )
            temp_model.append(
                keras.models.load_model(os.path.join(directory, path, "decoder"))
            )

            models.append(tuple(temp_model))

            if len(models) == new.no_models:
                break


        new.models = models
        new._assert_no_models()

        return new

    def __init__(
        self,
        no_models=3,
        threshold=0.66,
        name=None,
    ):
        self.no_models = no_models
        self.models = [self.seq2seq_model() for i in range(self.no_models)]
        self.threshold = threshold

        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = name if name else f"DeepEnsemble_{now}"

        self._assert_no_models()

    def _assert_no_models(self):
        assert self.no_models == len(
            self.models
        ), f"no_models attribute is {self.no_models}, but actual number of models is {len(self.models)}"

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
        if kwargs.get("verbose") is None:
            kwargs["verbose"] = 1
        for i in range(self.no_models):
            print(f"Training model no. {i+1}")

            encoder_input_data, decoder_input_data = kwargs.get("x")[0], kwargs.get("x")[1]
            decoder_target_data = kwargs.get("y")

            encoder_input_data, decoder_input_data, decoder_target_data = shuffle_matrices_by_row(encoder_input_data, decoder_input_data, decoder_target_data)

            kwargs["x"] = [encoder_input_data, decoder_input_data]
            kwargs["y"] = decoder_target_data

            self.models[i][0].fit(**kwargs)

    def predict(self, x, no_models, treshold):
        self._assert_no_models()

        # Encode the sequence for the models
        input_seq = self.encode_for_inference(x)

        # No. of remaining models to run
        remaining = no_models

        # No. of matching predictions required
        k = treshold * no_models

        predictions = {}

        models = self.models

        for i in models[:no_models]:
            out = self.decode_sequence(i[1], i[2], input_seq)
            predictions[out] = predictions.get(out, 0) + 1
            remaining -= 1

            if max(predictions.values()) + remaining < k:
                return x

            elif max(predictions.values()) >= k:
                return max(predictions, key=predictions.get)

        return x

    def predict_per_model(self, x, model_id):
        self._assert_no_models()

        # Encode the sequence for the models
        input_seq = self.encode_for_inference(x)

        model = self.models[model_id]

        out = self.decode_sequence(model[1], model[2], input_seq)

        return out
        
    def save(self, save_dir=None):
        self._assert_no_models()

        if save_dir is None:
            save_dir = self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        folder_names = [
            os.path.join(save_dir, f"model_{i+1}") for i in range(self.no_models)
        ]

        for i, path in enumerate(folder_names):
            if not os.path.exists(path):
                os.makedirs(path)

            self.models[i][0].save(os.path.join(path, "training"))
            self.models[i][1].save(os.path.join(path, "encoder"))
            self.models[i][2].save(os.path.join(path, "decoder"))

    def quantize(self, include_full_model=False):
        self.models = [list(i) for i in self.models]
        if include_full_model:
            model_range = range(0, 3)
        else:
            model_range = range(1, 3)

        for i in range(self.no_models):
            for j in model_range:
                self.models[i][j] = self._quantize_model(self.models[i][j])

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

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens), dtype="float32")
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            s_1, s_2 = c, h

        return decoded_sentence