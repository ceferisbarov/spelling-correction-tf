from iterative_data import (
    combinations,
    decoder_inputs,
    decoder_targets,
    encoder_inputs,
    max_encoders,
    max_decoders,
    num_encoders,
    num_decoders,
    reverse_indices,
    target_indices,
)
import os

from models import DeepEnsemble

os.makedirs("models/iterative", exist_ok=True)

batch_size = 32
epochs = 20

no_models = 8
threshold = int(round(no_models * 2 / 3) / no_models * 100) / 100

zippy = zip(
    decoder_inputs,
    decoder_targets,
    encoder_inputs,
    max_encoders,
    max_decoders,
    num_encoders,
    num_decoders,
    reverse_indices,
    target_indices,
)

for n, data in enumerate(zippy):
    decoder_input_data = data[0]
    decoder_target_data = data[1]
    encoder_input_data = data[2]
    max_encoder_seq_length = data[3]
    max_decoder_seq_length = data[4]
    num_encoder_tokens = data[5]
    num_decoder_tokens = data[6]
    reverse_target_char_index = data[7]
    target_token_index = data[8]

    de = DeepEnsemble(
        no_models=no_models,
        threshold=threshold,
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

    save_path = f"models/iterative/DE_v2_{n}"

    if not os.path.exists(save_path):
        de.save(save_path)
