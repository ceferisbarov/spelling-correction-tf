import time
from sklearn.metrics import accuracy_score
import numpy as np
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
from base import Model
from utils import plot_results, CER, WER
from tqdm import tqdm

chars = reverse_target_char_index.values()
test_data.dropna(inplace=True)
# The line below is used to use a fraction of the test dataset, mostly during debugging
# Set the frac argument to 1 to retrieve the complete dataset
# Or simply comment out the line
# test_data = test_data.sample(frac=0.1)
test_data = test_data[test_data["text"].apply(lambda s: all(c in chars for c in s))]
test_data = test_data[test_data["text"].str.len() <= train_data["text"].str.len().max()]

load_path = f"models/base_v2"

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

output = []
start = time.time()
for i in tqdm(range(len(test_data["text"])), desc=f"model_id={id}"):
    pred = model.predict(test_data["text"].iloc[i])
    output.append(pred.strip(" \n\r\t"))

end = time.time()
duration = end - start
latency = round(duration / len(test_data), 3)

test_data["prediction"] = output

wer = round(WER(y_true=test_data.label, y_pred=output) * 100, 2)
cer = round(CER(y_true=test_data.label, y_pred=output) * 100, 2)
acc = f"WER={wer} CER={cer}"

plot_results(data=test_data, method="base", index=id, accuracy=acc, latency=latency)
