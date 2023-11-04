import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import seaborn as sns

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


def calculate_distance(data):
    """
    Takes a dataframe that includes `text`, `label`, `prediction`, `distance`, and `org` columns.
    Plots the relationship between them.
    """
    data["distance"] = [
        levenshteinDistanceDP(i, t)
        for i, t in zip(list(data["label"]), list(data["prediction"]))
    ]
    data["org"] = [
        levenshteinDistanceDP(i, t)
        for i, t in zip(list(data["label"]), list(data["text"]))
    ]

    return data


def calculate_points(data):
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
    
    return np.array([org_points, pred_points])


def plot_results(data, no_models, treshold, accuracy, latency):

    now = datetime.now()

    stamp = f"{no_models}-{treshold}_{now.strftime('%Y%m%d-%H%M%S')}"

    data = calculate_distance(data)

    data.to_csv(f"results/DataFrame_{stamp}.csv")

    with open("results_ensemble.csv", 'a') as csv_file:
        csv_file.write(f"\n{no_models},{treshold},{accuracy},{str(latency)} sec,{data[data.distance == 0].shape[0]},{data[data.distance == 1].shape[0]},{data[data.distance == 2].shape[0]},{data[data.distance == 3].shape[0]},{data[data.distance == 4].shape[0]}")

    points = calculate_points(data)
    pd.DataFrame(points).to_csv(f"results/Points_{stamp}.csv")

    org_points, pred_points = points

    sns.set(font='Times New Roman', font_scale=0.7)
    sns.set_style("whitegrid")

    plt.plot(org_points * 100, label = "Word-Label", color='blue', linewidth=2, marker='o')
    plt.plot(pred_points * 100, label = "Prediction-Label", color='green', linewidth=2, marker='o')
    plt.fill_between(range(0, 4), org_points * 100, pred_points * 100, color="grey", alpha=0.3)

    x_ticks = np.arange(0, 4, 1)
    y_ticks = np.arange(0, 101, 10)
    y_tick_labels = ['{}%'.format(y) for y in y_ticks]

    plt.xticks(x_ticks)
    plt.yticks(y_ticks, y_tick_labels)

    plt.xlabel('The Levenshtein distance (cumulative)')
    plt.ylabel('Word Error Rate')
    plt.title('Model Performance')

    plt.legend()
    plt.savefig(f"results/Plot_{stamp}.jpg")
    plt.close()

    plot_splitted_dataset(data, stamp)


def shuffle_matrices_by_row(A, B, C):

    num_rows = A.shape[0]

    shuffled_indices = np.arange(num_rows)
    np.random.shuffle(shuffled_indices)

    A = A[shuffled_indices]
    B = B[shuffled_indices]
    C = C[shuffled_indices]

    return A, B, C


def split_dataset(data):
    return [data[data.org==0], data[data.org!=0]]


def plot_splitted_dataset(data, stamp):
    data_correct, data_incorrect = split_dataset(data)

    org_points_correct, pred_points_correct = calculate_points(data_correct)
    org_points_incorrect, pred_points_incorrect = calculate_points(data_incorrect)

    sns.set(font='Times New Roman', font_scale=0.7)
    sns.set_style("whitegrid")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(org_points_correct * 100, label = "Word-Label", color='green', linewidth=2, linestyle=':')
    axs[0].plot(pred_points_correct * 100, label = "Prediction-Label", color='green', linewidth=2, marker='o')
    axs[0].fill_between(range(0, 4), org_points_correct * 100, pred_points_correct * 100, color="grey", alpha=0.3)

    axs[1].plot(org_points_incorrect * 100, label = "Word-Label", color='blue', linewidth=2, linestyle=':')
    axs[1].plot(pred_points_incorrect * 100, label = "Prediction-Label", color='blue', linewidth=2, marker='o')
    axs[1].fill_between(range(0, 4), org_points_incorrect * 100, pred_points_incorrect * 100, color="grey", alpha=0.3)

    x_ticks = np.arange(0, 4, 1)
    y_ticks = np.arange(0, 101, 10)
    y_tick_labels = ['{}%'.format(y) for y in y_ticks]

    for i in range(2):
        axs[i].set_xticks(x_ticks)
        axs[i].set_yticks(y_ticks)
        axs[i].set_yticklabels(y_tick_labels) 

        axs[i].set_xlabel('The Levenshtein distance (cumulative)')
        axs[i].set_ylabel('Word Error Rate')
        axs[i].legend()

    axs[0].set_title('Originally Correct Dataset')
    axs[1].set_title('Originally Incorrect Dataset')

    plt.tight_layout()
    plt.savefig(f"results/Corr_Incorr_Plot_{stamp}.jpg")
    plt.close()
