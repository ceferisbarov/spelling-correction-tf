import numpy as np
import matplotlib.pyplot as plt


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
    print(list(map(lambda x: round(x, 3), org_points)))
    print(list(map(lambda x: round(x, 3), pred_points)))
    x = range(4)

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax3 = ax2.twinx()
    ax1.plot(x, org_points, "g-")
    ax2.plot(x, pred_points, "b-")
    ax3.plot(x, np.array(pred_points) - np.array(org_points), "r--")

    ax1.set_xlabel("X data")
    ax1.set_ylabel("Y1 data", color="g")
    ax2.set_ylabel("Y2 data", color="b")
    ax3.set_ylabel("Y3 data", color="r")

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0, 1)

    plt.savefig("images/results")
