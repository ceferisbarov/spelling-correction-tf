import time
import pandas as pd
from sklearn.metrics import accuracy_score

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
    input_indices,
    dfs,
    test_data,
)

from models import Model
from utils import plot_results, CER, WER
from tqdm import tqdm


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
    input_indices,
    dfs,
)

for i, data in enumerate(zippy):
    load_path = f"models/iterative/DE_v2_{i}/model_1"
    parameters = pd.read_csv("scripts/parameters.csv", header=0)
    decoder_input_data = data[0]
    decoder_target_data = data[1]
    encoder_input_data = data[2]
    max_encoder_seq_length = data[3]
    max_decoder_seq_length = data[4]
    num_encoder_tokens = data[5]
    num_decoder_tokens = data[6]
    reverse_target_char_index = data[7]
    target_token_index = data[8]
    input_token_index = data[9]
    train_data = data[10]

    chars = reverse_target_char_index.values()
    test_data.dropna(inplace=True)
    # The line below is used to use a fraction of the test dataset, mostly during debugging
    # Set the frac argument to 1 to retrieve the complete dataset
    # Or simply comment out the line
    # test_data = test_data.sample(frac=0.001)
    test_data = test_data[test_data["text"].apply(lambda s: all(c in chars for c in s))]
    test_data = test_data[
        test_data["text"].str.len() <= train_data["text"].str.len().max()
    ]

    myde = Model.load_from_dir(
        directory=load_path,
        max_encoder_seq_length=max_encoder_seq_length,
        max_decoder_seq_length=max_decoder_seq_length,
        num_encoder_tokens=num_encoder_tokens,
        num_decoder_tokens=num_decoder_tokens,
        reverse_target_char_index=reverse_target_char_index,
        target_token_index=target_token_index,
        input_token_index=input_token_index,
    )
    myde.quantize()

    output = []
    start = time.time()
    for i in tqdm(range(len(test_data["text"])), desc=f"model_id={id}"):
        pred = myde.predict(test_data["text"].iloc[i])
        output.append(pred.strip(" \n\r\t"))

    end = time.time()
    duration = end - start
    latency = round(duration / len(test_data), 3)

    test_data["prediction"] = output

    wer = round(WER(y_true=test_data.label, y_pred=output) * 100, 2)
    cer = round(CER(y_true=test_data.label, y_pred=output) * 100, 2)
    acc = f"WER={wer} CER={cer}"

    plot_results(data=test_data, method="base", index=id, accuracy=acc, latency=latency)
