import os
from datetime import datetime

import numpy as np
from tensorflow import keras

from scipy.stats import entropy

from utils import shuffle_matrices_by_row
from base import Model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

class DeltaModel(Model):
    """
    A single Seq2Seq model. Difference between first and second
    top softmax scores is used as a metric of uncertainty estimation.
    """

    @staticmethod
    def load_from_dir(directory, **kwargs):
        """
        Static method to load model from the given directory.
        The model should be saved in SaveModel format
        """
        new = DeltaModel(**kwargs)
        model = []
        model.append(keras.models.load_model(os.path.join(directory, "training")))
        model.append(keras.models.load_model(os.path.join(directory, "encoder")))
        model.append(keras.models.load_model(os.path.join(directory, "decoder")))

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
        self.name = name if name else f"Delta_{now}"

    def predict(self, x, threshold=None):
        if not threshold:
            threshold = self.threshold

        input_seq = self.encode_for_inference(x)

        word, vectors = self.decode_sequence(self.model[1], self.model[2], input_seq)
        delta = 0
        for vector in vectors:
            temp = vector.flatten()
            temp = np.sort(temp)
            delta += temp[-1].item() - temp[-2].item()

        delta = delta / (len(vectors))
        
        return word if delta >= threshold else x


class EntropyModel(DeltaModel):
    """
    A single Seq2Seq model. Entropy of softmax scores
    is used as a metric of uncertainty estimation.
    """

    @staticmethod
    def load_from_dir(directory, **kwargs):
        """
        Static method to load model from the given directory.
        The model should be saved in SaveModel format
        """
        new = EntropyModel(**kwargs)
        model = []
        model.append(keras.models.load_model(os.path.join(directory, "training")))
        model.append(keras.models.load_model(os.path.join(directory, "encoder")))
        model.append(keras.models.load_model(os.path.join(directory, "decoder")))

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
        self.name = name if name else f"Entropy_{now}"

    def predict(self, x, threshold=None):
        if not threshold:
            threshold = self.threshold

        input_seq = self.encode_for_inference(x)

        word, vectors = self.decode_sequence(self.model[1], self.model[2], input_seq)
        total_entropy = 0
        for vector in vectors:
            temp = vector.flatten()
            total_entropy += entropy(temp)
        avg_entropy = total_entropy / (len(vectors))
        # print(avg_entropy)
        return word if avg_entropy <= threshold else x



class DeepEnsemble(Model):
    """Deep Ensemble class for Seq2Seq ensemble models."""

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

    def fit(self, **kwargs):
        if kwargs.get("verbose") is None:
            kwargs["verbose"] = 1
        for i in range(self.no_models):
            print(f"Training model no. {i+1}")

            encoder_input_data, decoder_input_data = (
                kwargs.get("x")[0],
                kwargs.get("x")[1],
            )
            decoder_target_data = kwargs.get("y")

            (
                encoder_input_data,
                decoder_input_data,
                decoder_target_data,
            ) = shuffle_matrices_by_row(
                encoder_input_data, decoder_input_data, decoder_target_data
            )

            kwargs["x"] = [encoder_input_data, decoder_input_data]
            kwargs["y"] = decoder_target_data

            log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            kwargs["callbacks"] = [tensorboard_callback, EarlyStopping(monitor="val_accuracy", patience=3)]

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
            out, _ = self.decode_sequence(i[1], i[2], input_seq)
            predictions[out] = predictions.get(out, 0) + 1
            remaining -= 1

            if max(predictions.values()) + remaining < k:
                return x

            elif max(predictions.values()) >= k:
                return max(predictions, key=predictions.get)

        return x

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
