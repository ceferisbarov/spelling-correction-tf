from load_data import decoder_input_data, decoder_target_data, encoder_input_data
import os

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from load_data import (
    decoder_input_data,
    decoder_target_data,
    encoder_input_data,
    max_encoder_seq_length,
    max_decoder_seq_length,
    num_encoder_tokens,
    num_decoder_tokens,
    reverse_target_char_index,
    target_token_index,
)
from models import Model

batch_size = 32
epochs = 20

de = Model(
    max_encoder_seq_length=max_encoder_seq_length,
    max_decoder_seq_length=max_decoder_seq_length,
    num_encoder_tokens=num_encoder_tokens,
    num_decoder_tokens=num_decoder_tokens,
    reverse_target_char_index=reverse_target_char_index,
    target_token_index=target_token_index,
)

history = de.fit(
    x=[encoder_input_data, decoder_input_data],
    y=decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)

save_path = "models/base_v2"

if not os.path.exists(save_path):
    de.save(save_path)
