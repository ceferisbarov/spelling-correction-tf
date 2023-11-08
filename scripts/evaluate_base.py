import time
from sklearn.metrics import accuracy_score
import numpy as np
from load_data import (
    reverse_target_char_index,
    test_data,
    train_data,
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

for id in range(1, 9):
    load_path = f"models/DE_v4/model_{id}"

    myde = Model.load_from_dir(load_path)
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

    wer = round(WER(y_true=test_data.label, y_pred=output)*100, 2)
    cer = round(CER(y_true=test_data.label, y_pred=output)*100, 2)
    acc = f"WER={wer} CER={cer}"

    plot_results(data=test_data, method="base", index=id, accuracy=acc, latency=latency)