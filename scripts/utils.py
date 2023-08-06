import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd


def levenshteinDistanceDP(token1, token2):
    """
    Takes two strings as input and returns the Levenshtein distance as float.
    """
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


def plot_results(data):
    """
    Takes a dataframe that includes `text`, `label`, `prediction`, `distance`, and `org` columns.
    Plots the relationship between them.
    """
    data["distance"] = [
        levenshteinDistanceDP(i, t)
        for i, t in zip(list(data["label"]), list(data["prediction"]))
    ]
    data[data["distance"] == 0].shape[0] / data.shape[0]
    data["org"] = [
        levenshteinDistanceDP(i, t)
        for i, t in zip(list(data["label"]), list(data["text"]))
    ]

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    data.to_csv(f"results/DataFrame_{date_time}.csv")

    length = data.shape[0]
    org_points = [
        data[data["org"] == 0].shape[0] / length,
        data[data["org"] <= 1].shape[0] / length,
        data[data["org"] <= 2].shape[0] / length,
        data[data["org"] <= 3].shape[0] / length,
    ]
    pred_points = [
        data[data["distance"] == 0].shape[0] / length,
        data[data["distance"] <= 1].shape[0] / length,
        data[data["distance"] <= 2].shape[0] / length,
        data[data["distance"] <= 3].shape[0] / length,
    ]
    
    points = np.array([org_points, pred_points])
    pd.DataFrame(points).to_csv(f"results/Points_{date_time}.csv")
    org_points, pred_points = points[0], points[1]

    plt.plot(org_points * 100, label = "Word-Label", color='blue', linewidth=2, marker='o')
    plt.plot(pred_points * 100, label = "Prediction-Label", color='green', linewidth=2, marker='o')
    plt.plot((pred_points - org_points) * 100, label = "Improvement", color='red', linewidth=2, linestyle=':')

    x_ticks = np.arange(0, 4, 1)
    y_ticks = np.arange(0, 101, 10)
    y_tick_labels = ['{}%'.format(y) for y in y_ticks]

    plt.xticks(x_ticks)
    plt.yticks(y_ticks, y_tick_labels)

    plt.xlabel('The Levenshtein distance (cumulative)')
    plt.ylabel('Word Error Rate')
    plt.title('Three Lines Plot')

    plt.legend()
    plt.savefig(f"results/Plot_{date_time}")

def shuffle_matrices_by_row(A, B, C):

    num_rows = A.shape[0]

    shuffled_indices = np.arange(num_rows)
    np.random.shuffle(shuffled_indices)

    A = A[shuffled_indices]
    B = B[shuffled_indices]
    C = C[shuffled_indices]

    return A, B, C
