import pandas as pd

punct = "()\/\\:+~!;*?"

df = pd.read_csv("data/raw.csv")
print(df.shape)
df = df.drop(["id"], axis=1)

df.dropna(inplace=True)

def is_abnormal(text):
    return any([i in text for i in punct])

df["abnormal"] = df["text"].apply(is_abnormal) | df["label"].apply(is_abnormal)

print(df.head())
df = df[df["abnormal"] == False]
print(df.head())
df = df.drop(["abnormal"], axis=1)

print(df.head())

print(df.shape)

df.to_csv("data/processed.csv", index=False)