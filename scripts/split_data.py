import pandas as pd
from sklearn.model_selection import train_test_split

test_size = 0.2

df = pd.read_csv("data/processed.csv")
df.dropna(inplace=True)

train_data, test_data = train_test_split(df, test_size=test_size, random_state=1)

train_data.to_csv("data/train.csv", index=False, header=True)
test_data.to_csv("data/test.csv", index=False, header=True)
