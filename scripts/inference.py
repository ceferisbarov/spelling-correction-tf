from models import EntropyModel
from base import Model
from load_data import (
    reverse_target_char_index,
    test_data,
    train_data,
    max_encoder_seq_length,
    max_decoder_seq_length,
    num_encoder_tokens,
    num_decoder_tokens,
    target_token_index,
    input_token_index,
)

load_path = "models/base_v1"

threshold = 0.8
model = Model.load_from_dir(
    directory=load_path,
    max_encoder_seq_length=max_encoder_seq_length,
    max_decoder_seq_length=max_decoder_seq_length,
    num_encoder_tokens=num_encoder_tokens,
    num_decoder_tokens=num_decoder_tokens,
    reverse_target_char_index=reverse_target_char_index,
    target_token_index=target_token_index,
    input_token_index=input_token_index,
)

# model.quantize()

for i in ["salim", "necesn", "komputer", "telefin", "piyanina"]:
    print(model.predict(i))
