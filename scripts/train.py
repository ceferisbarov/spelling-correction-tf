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

save_path = "models/DE_v1"
batch_size = 32
epochs = 10

de = DeepEnsemble(no_models=5, threshold=0.8)
callbacks = [EarlyStopping(monitor="val_accuracy", patience=5)]

plot_model(de.models[0][0], show_shapes=True, to_file="images/model.png")
plot_model(de.models[0][1], show_shapes=True, to_file="images/encoder.png")
plot_model(de.models[0][2], show_shapes=True, to_file="images/decoder.png")

with mlflow.start_run(run_name="setting-up"):
    history = de.fit(
        x=[encoder_input_data, decoder_input_data],
        y=decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=callbacks,
    )

    if not os.path.exists(save_path):
        de.save(save_path)
