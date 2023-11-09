import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

test_size = 0.1

df = pd.read_csv("data/processed.csv")
df.dropna(inplace=True)

train_data, test_data = train_test_split(df, test_size=test_size, random_state=1)

correct = train_data[train_data["text"] == train_data["label"]]
incorrect = train_data[train_data["text"] != train_data["label"]]

maximum = min(correct.shape[0], incorrect.shape[0])

combinations = (
    (maximum, 0),
    (0.75 * maximum, 0.25 * maximum),
    (0.5 * maximum, 0.5 * maximum),
    (0.25 * maximum, 0.75 * maximum),
    (0, maximum),
)

dfs = []

decoder_inputs = []
decoder_targets = []
encoder_inputs = []
max_encoders = []
max_decoders = []
num_encoders = []
num_decoders = []
reverse_indices = []
target_indices = []
input_indices = []

for n, comb in enumerate(combinations):
    i, j = int(comb[0]), int(comb[1])
    temp_correct = correct.sample(n=i, random_state=69)
    temp_incorrect = incorrect.sample(n=j, random_state=69)
    temp_train = pd.concat([temp_correct, temp_incorrect])

    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()

    for idx, row in temp_train.iterrows():
        input_text = row[0]
        target_text = row[1]
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = "\t" + target_text + "\n"
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    input_characters.insert(0, " ")
    target_characters = sorted(list(target_characters))
    target_characters.insert(0, " ")

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    num_encoders.append(num_encoder_tokens)
    num_decoders.append(num_decoder_tokens)
    max_encoders.append(max_encoder_seq_length)
    max_decoders.append(max_decoder_seq_length)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype="float32",
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0

    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items()
    )

    reverse_indices.append(reverse_target_char_index)
    target_indices.append(target_token_index)
    decoder_inputs.append(decoder_input_data)
    decoder_targets.append(decoder_target_data)
    encoder_inputs.append(encoder_input_data)
    input_indices.append(input_token_index)
    dfs.append(temp_train)
    with open(f"data/input_token_index_{n}.json", "w") as fp:
        json.dump(input_token_index, fp)

    with open(f"data/target_token_index.json_{n}", "w") as fp:
        json.dump(target_token_index, fp)

    print(f"Batch {n} ready.")

print("Finished loading the data.")
