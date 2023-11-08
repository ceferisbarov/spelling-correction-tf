import time
from sklearn.metrics import accuracy_score
import numpy as np
from load_data import (
    reverse_target_char_index,
    test_data,
    train_data,
)
from models import DeltaModel
from utils import plot_results, CER, WER
from tqdm import tqdm


chars = reverse_target_char_index.values()
test_data.dropna(inplace=True)
# The line below is used to use a fraction of the test dataset, mostly during debugging
# Set the frac argument to 1 to retrieve the complete dataset
# Or simply comment out the line
# test_data = test_data.sample(frac=1)
test_data = test_data[test_data["text"].apply(lambda s: all(c in chars for c in s))]
test_data = test_data[test_data["text"].str.len() <= train_data["text"].str.len().max()]


for id in [1, 2, 6, 7]:
    threshold_range = np.linspace(0.8, 0.9, 5)

    load_path = f"models/DE_v4/model_{id}"
    myde = DeltaModel.load_from_dir(load_path)
    myde.quantize()

    for threshold in threshold_range:

        output = []
        start = time.time()

        for i in tqdm(range(len(test_data["text"])), desc=f"m={id}, t={threshold}"):
            pred = myde.predict(test_data["text"].iloc[i], threshold=threshold)
            output.append(pred.strip(" \n\r\t"))

        end = time.time()
        duration = end - start
        latency = round(duration / len(test_data), 3)

        test_data["prediction"] = output

        wer = round(WER(y_true=test_data.label, y_pred=output)*100, 2)
        cer = round(CER(y_true=test_data.label, y_pred=output)*100, 2)
        acc = f"WER={wer} CER={cer}"

        plot_results(data=test_data, method="delta", accuracy=acc, latency=latency, index=id, treshold=threshold)
