from load_data import decoder_input_data, decoder_target_data, encoder_input_data
print(decoder_input_data.shape)
print(decoder_target_data.shape)
print(encoder_input_data.shape)

import os
import json

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from load_data import decoder_input_data, decoder_target_data, encoder_input_data
from models import DeepEnsemble

import mlflow

mlflow.set_tracking_uri("http://10.20.36.26:5002")
mlflow.set_experiment("deep-ensembles")
mlflow.autolog()

batch_size = 32
epochs = 10

no_models = 1
threshold = int(round(no_models * 2 / 3) / no_models * 100) / 100

de = DeepEnsemble(no_models=no_models, threshold=threshold)
callbacks = [EarlyStopping(monitor="val_accuracy", patience=5)]

plot_model(de.models[0][0], show_shapes=True, to_file="images/model.png")
plot_model(de.models[0][1], show_shapes=True, to_file="images/encoder.png")
plot_model(de.models[0][2], show_shapes=True, to_file="images/decoder.png")

history = de.fit(
    x=[encoder_input_data, decoder_input_data],
    y=decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=callbacks,
)

save_path = "models/DE_v4"

if not os.path.exists(save_path):
    de.save(save_path)
