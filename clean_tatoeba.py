import pandas as pd
from langdetect import detect
from tqdm import tqdm

INPUT_FILE = "data/tatoeba/tatoeba-sentpairs.tsv"
OUTPUT_FILE = "data/tatoeba/cleaned_en_hi.csv"

chunksize = 100_000

def is_valid_pair(src, tgt):
    try:
        return detect(src) == "en" and detect(tgt) == "hi"
    except:
        return False

print("🔹 Starting cleaning process...")

cleaned_chunks = []

for chunk in tqdm(pd.read_csv(
        INPUT_FILE,
        sep="\t",
        names=["source", "target"],
        chunksize=chunksize,
        encoding="utf-8",
        on_bad_lines="skip"
    )):

    chunk = chunk.dropna()
    chunk = chunk[
        (chunk["source"].str.strip() != "") &
        (chunk["target"].str.strip() != "")
    ]

    chunk = chunk.drop_duplicates()

    mask = chunk.apply(lambda x: is_valid_pair(x["source"], x["target"]), axis=1)
    chunk = chunk[mask]

    cleaned_chunks.append(chunk)

final_df = pd.concat(cleaned_chunks)
final_df.to_csv(OUTPUT_FILE, index=False)

print("✅ Cleaning complete!")
print("Saved to:", OUTPUT_FILE)
