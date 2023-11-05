import time
import pandas as pd
from sklearn.metrics import accuracy_score

from load_data import (
    reverse_target_char_index,
    test_data,
    train_data,
)
from models import DeepEnsemble
from utils import plot_results
from tqdm import tqdm

chars = reverse_target_char_index.values()
test_data.dropna(inplace=True)
# The line below is used to use a fraction of the test dataset, mostly during debugging
# Set the frac argument to 1 to retrieve the complete dataset
# Or simply comment out the line
# test_data = test_data.sample(frac=0.001)
test_data = test_data[test_data["text"].apply(lambda s: all(c in chars for c in s))]
test_data = test_data[test_data["text"].str.len() <= train_data["text"].str.len().max()]

load_path = "models/DE_v3"
parameters = pd.read_csv("scripts/parameters.csv", header=0)

myde = DeepEnsemble.load_from_dir(load_path, no_models=8, threshold=8)
myde.quantize()

for n, params in parameters.iterrows():
    no_models = int(params.n_models)
    treshold = int(params.treshold / no_models * 100) / 100

    output = []
    start = time.time()
    for i, row in enumerate(test_data["text"]):
        pred = myde.predict(row, no_models=no_models, treshold=treshold)
        output.append(pred.strip(" \n\r\t"))
        if i % 25 == 0:
            print(i)

    end = time.time()
    duration = end - start
    latency = round(duration / len(test_data), 3)

    test_data["prediction"] = output
    accuracy = round(
        accuracy_score(y_true=test_data.label, y_pred=test_data.prediction), 3
    )

    plot_results(
        test_data, int(params.n_models), int(params.treshold), accuracy, latency
    )
