import pandas as pd

df = pd.read_csv("data/tatoeba/cleaned_en_hi.csv")
print("Shape:", df.shape)
print(df.sample(5))
