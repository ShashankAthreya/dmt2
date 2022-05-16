import pandas as pd
from sklearn.utils import shuffle

# read data
# working with a small sample..
# shuffling the data, then taking out a sample of 500k
data = pd.read_csv('./data/training_set_VU_DM.csv', nrows=500000)
df = pd.DataFrame(data)
df = df.sample(frac=1)

df = df[:500000]

df.to_csv('500k_csv_data.csv')

print(df)
