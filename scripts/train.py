from load_data import decoder_input_data, decoder_target_data, encoder_input_data
import os

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from load_data import decoder_input_data, decoder_target_data, encoder_input_data
from models import DeepEnsemble

batch_size = 32
epochs = 20

no_models = 8
threshold = int(round(no_models * 2 / 3) / no_models * 100) / 100

de = DeepEnsemble(no_models=no_models, threshold=threshold)

plot_model(de.models[0][0], show_shapes=True, to_file="images/model.png")
plot_model(de.models[0][1], show_shapes=True, to_file="images/encoder.png")
plot_model(de.models[0][2], show_shapes=True, to_file="images/decoder.png")

history = de.fit(
    x=[encoder_input_data, decoder_input_data],
    y=decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2)

save_path = "models/DE_v4"

if not os.path.exists(save_path):
    de.save(save_path)
