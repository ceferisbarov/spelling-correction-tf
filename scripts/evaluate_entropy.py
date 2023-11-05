import time
from sklearn.metrics import accuracy_score
import numpy as np
from load_data import (
    reverse_target_char_index,
    test_data,
    train_data,
)
from models import EntropyModel
from utils import plot_results

threshold_range = np.linspace(0.2, 0.5, 7)

chars = reverse_target_char_index.values()
test_data.dropna(inplace=True)
# The line below is used to use a fraction of the test dataset, mostly during debugging
# Set the frac argument to 1 to retrieve the complete dataset
# Or simply comment out the line
# test_data = test_data.sample(frac=1)
test_data = test_data[test_data["text"].apply(lambda s: all(c in chars for c in s))]
test_data = test_data[test_data["text"].str.len() <= train_data["text"].str.len().max()]

for i in range(1, 9):
    load_path = f"models/DE_v3/model_{i}"
    no_models = 1
    threshold = 0.80

    myde = EntropyModel.load_from_dir(load_path, threshold=threshold)
    myde.quantize()

    output = []
    start = time.time()
    for i, row in enumerate(test_data["text"]):
        pred = myde.predict(row, certain=False)
        output.append(pred.strip(" \n\r\t"))
        # if i % 25 == 0:
        #     print(i)

    end = time.time()
    duration = end - start
    latency = round(duration / len(test_data), 3)

    accuracy = round(accuracy_score(y_true=test_data.label, y_pred=output), 5)
    print(no_models, threshold, accuracy, latency, "uncertain")

    for threshold in threshold_range:
        print(threshold)
        output = []
        start = time.time()
        for i, row in enumerate(test_data["text"]):
            pred = myde.predict(row, certain=True, threshold=threshold)
            output.append(pred.strip(" \n\r\t"))
            # if i % 25 == 0:
            #     print(i)

        end = time.time()
        duration = end - start
        latency = round(duration / len(test_data), 3)
        accuracy = round(accuracy_score(y_true=test_data.label, y_pred=output), 5)
        print(no_models, threshold, accuracy, latency, "certain")
